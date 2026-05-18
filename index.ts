import {
	appendFileSync,
	mkdirSync,
	existsSync,
} from "node:fs";
import { join, dirname } from "node:path";

import type {
	ExtensionAPI,
	ExtensionContext,
	ProviderModelConfig,
	ThemeColor,
} from "@earendil-works/pi-coding-agent";

const PROVIDER_NAME = "openai-compat";

// ---------------------------------------------------------------------------
// Model discovery (unchanged from upstream)
// ---------------------------------------------------------------------------

interface OpenAIModelEntry {
	id: string;
	owned_by?: string;
	meta?: { n_ctx_train?: number };
	max_tokens?: number;
	max_input_tokens?: number;
	context_window?: number;
	supportsVision?: boolean;
}

interface LiteLLMModelInfo {
	model_name: string;
	model_info?: {
		max_tokens?: number | null;
		max_input_tokens?: number | null;
		supports_vision?: boolean | null;
	};
}

interface LiteLLMModelInfoResponse {
	data?: LiteLLMModelInfo[];
}

interface OpenAIModelsResponse {
	data?: OpenAIModelEntry[];
	models?: OpenAIModelEntry[];
}

const KNOWN_CONTEXT_WINDOWS: Array<[RegExp, number]> = [
	[/qwen3/i, 131072],
	[/qwq-32b/i, 131072],
	[/deepseek-r1/i, 131072],
];

const REASONING_MODEL_PATTERNS = [
	/qwen3/i, /qwq/i, /deepseek-r1/i, /deepseek-reasoner/i,
];

const STREAMING_USAGE_ENABLED = process.env.OPENAI_COMPAT_NO_STREAMING_USAGE !== "1";

function resolveBaseUrl(): string | undefined {
	return process.env.OPENAI_COMPAT_BASE_URL ?? process.env.OPENAI_BASE_URL;
}

function resolveApiKey(): string | undefined {
	return process.env.OPENAI_COMPAT_API_KEY ?? process.env.OPENAI_API_KEY;
}

async function fetchJson<T>(url: string, headers: Record<string, string>): Promise<T | null> {
	try {
		const response = await fetch(url, { signal: AbortSignal.timeout(5000), headers });
		if (!response.ok) return null;
		return (await response.json()) as T;
	} catch {
		return null;
	}
}

async function discoverModels(baseUrl: string, apiKey?: string): Promise<OpenAIModelEntry[]> {
	const base = baseUrl.replace(/\/+$/, "").replace(/\/v1$/, "");
	const headers: Record<string, string> = { Accept: "application/json" };
	if (apiKey) headers["Authorization"] = `Bearer ${apiKey}`;

	const modelsBody = await fetchJson<OpenAIModelsResponse>(`${base}/v1/models`, headers);
	if (!modelsBody) throw new Error("Failed to fetch /v1/models");

	const entries = (modelsBody.data ?? modelsBody.models ?? []).filter((m) => m.id && m.id.length > 0);

	// LiteLLM /model/info for richer metadata.
	const infoBody = await fetchJson<LiteLLMModelInfoResponse>(`${base}/model/info`, headers);
	if (infoBody?.data) {
		const infoByName = new Map<string, LiteLLMModelInfo>();
		for (const item of infoBody.data) infoByName.set(item.model_name, item);
		for (const entry of entries) {
			const mi = infoByName.get(entry.id)?.model_info;
			if (!mi) continue;
			if (mi.max_tokens != null && mi.max_tokens > 0) entry.context_window = mi.max_tokens;
			if (mi.max_input_tokens != null && mi.max_input_tokens > 0) entry.max_input_tokens = mi.max_input_tokens;
			if (mi.supports_vision != null) entry.supportsVision = mi.supports_vision;
		}
	}
	return entries;
}

function resolveModelId(entry: OpenAIModelEntry): string {
	const slashIdx = entry.id.lastIndexOf("/");
	return slashIdx !== -1 ? entry.id.slice(slashIdx + 1) : entry.id;
}

function isReasoningModel(modelId: string): boolean {
	return REASONING_MODEL_PATTERNS.some((p) => p.test(modelId));
}

function resolveContextWindow(entry: OpenAIModelEntry): number {
	const fromApi = entry.context_window ?? entry.max_input_tokens ?? entry.max_tokens ?? entry.meta?.n_ctx_train;
	if (fromApi && fromApi > 0) return fromApi;
	for (const [pattern, size] of KNOWN_CONTEXT_WINDOWS) {
		if (pattern.test(entry.id)) return size;
	}
	return 32768;
}

function toProviderModel(entry: OpenAIModelEntry): ProviderModelConfig {
	const contextWindow = resolveContextWindow(entry);
	const reasoning = isReasoningModel(entry.id);
	return {
		id: resolveModelId(entry),
		name: entry.id,
		reasoning,
		input: entry.supportsVision ? ["text", "image"] : ["text"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow,
		maxTokens: reasoning ? Math.min(16384, contextWindow) : Math.min(4096, contextWindow),
		compat: {
			maxTokensField: "max_tokens",
			supportsDeveloperRole: false,
			supportsReasoningEffort: false,
			supportsUsageInStreaming: STREAMING_USAGE_ENABLED,
			...(reasoning ? { thinkingFormat: "qwen-chat-template" } : {}),
		},
	};
}

// ---------------------------------------------------------------------------
// Footer state — quota + session stats shown in Pi status bar
// ---------------------------------------------------------------------------

interface UsageEntry { timestamp: number; totalTokens: number; }

/** Quota limits from LiteLLM /user/info. */
interface QuotaLimits {
	tokenLimit: number | null;   // tpm_limit
	rpmLimit: number | null;     // rpm_limit
}

/** Sliding window for RPM/TPM tracking. */
interface UsageWindow {
	windowStart: number;   // start of current 60s window
	requests: number;      // requests in current window
	entries: UsageEntry[]; // token-count entries in current window
}

/** Prune window entries older than 60s; reset counters if window expired. */
function pruneWindow(w: UsageWindow): void {
	const cutoff = Date.now() - TRACKING_WINDOW_MS;
	if (w.windowStart < cutoff) {
		w.windowStart = Date.now();
		w.requests = 0;
		w.entries.length = 0;
	}
	w.entries = w.entries.filter((e) => e.timestamp >= cutoff);
}

const TRACKING_WINDOW_MS = 60_000;
const windowState: UsageWindow = { windowStart: 0, requests: 0, entries: [] };

/** Call on request start to increment RPM counter. */
function trackRequestStart(): void {
	pruneWindow(windowState);
	windowState.requests++;
}

/** Call on response to record token count for TPM. */
function trackTokens(tokenCount: number): void {
	windowState.entries.push({ timestamp: Date.now(), totalTokens: tokenCount });
	pruneWindow(windowState);
}

/** Current RPM from sliding window. */
function getRollingRPM(): number {
	pruneWindow(windowState);
	return windowState.requests;
}

/** Current TPM from sliding window. */
function getRollingTPM(): number {
	pruneWindow(windowState);
	return windowState.entries.reduce((sum, e) => sum + e.totalTokens, 0);
}

/** Format a status-label string for RPM/TPM, with colour-coding per proximity to limit. */
function buildQuotaLabel(ctx: ExtensionContext): string | undefined {
	const rpm = getRollingRPM();
	const tpm = getRollingTPM();
	const parts: string[] = [];

	if (quota.rpmLimit !== null) {
		const rpmPct = rpm / quota.rpmLimit;
		const rpmColor = rpmPct > 0.85 ? "error" : rpmPct > 0.65 ? "warning" : "dim" as ThemeColor;
		parts.push(ctx.ui.theme.fg(rpmColor, `rps:${rpm}/${quota.rpmLimit}`));
	}
	if (quota.tokenLimit !== null) {
		const tpmPct = tpm / quota.tokenLimit;
		const tpmColor = tpmPct > 0.85 ? "error" : tpmPct > 0.65 ? "warning" : "dim" as ThemeColor;
		parts.push(ctx.ui.theme.fg(tpmColor, `${fmtTokens(tpm)}/${fmtTokens(quota.tokenLimit)} tpm`));
	}
	return parts.length > 0 ? parts.join(ctx.ui.theme.fg("dim", "  ")) : undefined;
}

/** Check RPM threshold and fire alerts if crossing 85%. */
function checkQuotaAlerts(ctx: ExtensionContext): void {
	if (quota.rpmLimit !== null) {
		const rpm = getRollingRPM();
		const rpmPct = rpm / quota.rpmLimit;
		const crossed85 = rpmPct > 0.85 && (lastNotifiedPct === null || lastNotifiedPct < 0.85);
		if (crossed85) {
			lastNotifiedPct = rpmPct;
			ctx.ui.notify(`⚠️ RPM approaching limit: ${rpm}/${quota.rpmLimit} requests/min (${Math.round(rpmPct * 100)}%). Slow down to avoid being cut off.`, "warning");
		}
	}
}

/** 429 detection state. */
interface QuotaRateLimited {
	isLimited: boolean;
	resetAt: string | null;
	retryAfterSecs: number | null;
}

const rateLimited: QuotaRateLimited = { isLimited: false, resetAt: null, retryAfterSecs: null };
const quota: QuotaLimits = { tokenLimit: null, rpmLimit: null };
let lastNotifiedPct: number | null = null;
let lastContextNotifiedPct: number | null = null;
let lastBudgetNotified: boolean = false; // pre-turn budget warning — fires once per session

function updateQuotaStatus(ctx: ExtensionContext): void {
	if (ctx.model?.provider !== PROVIDER_NAME) {
		ctx.ui.setStatus(QUOTA_STATUS_KEY, undefined);
		return;
	}

	if (rateLimited.isLimited) {
		const reset = rateLimited.resetAt
			? `reset ${fmtResetTime(rateLimited.resetAt)}`
			: rateLimited.retryAfterSecs
				? `retry in ${rateLimited.retryAfterSecs}s`
				: "rate limited";
		ctx.ui.setStatus(QUOTA_STATUS_KEY, ctx.ui.theme.fg("error", `⛔ ${reset}`));
		return;
	}

	const label = buildQuotaLabel(ctx);
	if (label) ctx.ui.setStatus(QUOTA_STATUS_KEY, label);
	checkQuotaAlerts(ctx);
}

// Session state
interface SessionStats {
	finishReason: string | null;
	contextUsed: number | null;
	contextWindow: number | null;
}
const session: SessionStats = { finishReason: null, contextUsed: null, contextWindow: null };

const logDir = "~/.pi/logs/pi-openai-compat";
const logFile = "session-log.jsonl";

function logEvent(ctx: ExtensionContext, evt: { ts: string; event: string; details: Record<string, unknown> }): void {
	try {
		const path = join(ctx.cwd, logDir, logFile);
		if (!existsSync(dirname(path))) {
			mkdirSync(dirname(path), { recursive: true });
		}
		appendFileSync(path, JSON.stringify(evt) + "\n", "utf8");
	} catch {
		/* best-effort only — never interrupt the model loop */
	}
}

const QUOTA_STATUS_KEY  = "openai-compat-quota";
const SESSION_STATUS_KEY = "openai-compat-session";
const TTFT_STATUS_KEY    = "openai-compat-ttft";

const FINISH_REASON_MAP: Record<string, { icon: string; label: string; color: string }> = {
	end_turn: { icon: "⏹", label: "done", color: "dim" },
	stop:     { icon: "⏹", label: "done", color: "dim" },
	stop_sequence: { icon: "⏹", label: "stop_seq", color: "dim" },
	max_tokens: { icon: "✂", label: "max_tokens", color: "warning" },
	length:     { icon: "✂", label: "max_tokens", color: "warning" },
	tool_use:   { icon: "🔧", label: "tool", color: "dim" },
	tool_calls: { icon: "🔧", label: "tool", color: "dim" },
	error:      { icon: "⚠", label: "error", color: "error" },
};

function fmtTokens(n: number): string {
	if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
	if (n >= 1_000) return `${Math.round(n / 1_000)}K`;
	return String(n);
}

function fmtResetTime(resetAt: string): string {
	const m = resetAt.match(/(\d{2}:\d{2}):\d{2} (UTC)/i);
	return m ? `${m[1]} ${m[2]}` : resetAt;
}

function updateSessionStatus(ctx: ExtensionContext): void {
	if (ctx.model?.provider !== PROVIDER_NAME) {
		ctx.ui.setStatus(SESSION_STATUS_KEY, undefined);
		return;
	}
	const parts: string[] = [];
	if (session.contextUsed !== null && session.contextWindow !== null && session.contextWindow > 0) {
		const pct = Math.round(session.contextUsed / session.contextWindow * 100);
		const color = pct > 60 ? "error" : pct > 40 ? "warning" : "dim";
		parts.push(ctx.ui.theme.fg(color, `${fmtTokens(session.contextUsed)}/${fmtTokens(session.contextWindow)} ctx ${pct}%`));
	}
	if (session.finishReason) {
		const def = FINISH_REASON_MAP[session.finishReason] ?? { icon: "?", label: session.finishReason, color: "dim" };
		parts.push(ctx.ui.theme.fg(def.color as ThemeColor, `${def.icon} ${def.label}`));
	}
	ctx.ui.setStatus(SESSION_STATUS_KEY, parts.length > 0 ? parts.join(ctx.ui.theme.fg("dim", "  ")) : undefined);
}

function resetSessionStats(): void {
	session.finishReason = null;
	session.contextUsed = null;
	session.contextWindow = null;
}

// ---------------------------------------------------------------------------
// TTFT (time-to-first-token) waiting indicator
// ---------------------------------------------------------------------------

interface WaitState {
	requestSentAt: number | null;
	firstTokenAt: number | null;
	gotFirstToken: boolean;
	timer: ReturnType<typeof setInterval> | null;
	ui: ExtensionContext["ui"] | null;
}
const wait: WaitState = { requestSentAt: null, firstTokenAt: null, gotFirstToken: false, timer: null, ui: null };

function clearWaitTimer(): void {
	if (wait.timer !== null) { clearInterval(wait.timer); wait.timer = null; }
}

function tickWaitStatus(): void {
	if (!wait.ui || !wait.requestSentAt || wait.gotFirstToken) return;
	wait.ui.setStatus(TTFT_STATUS_KEY, wait.ui.theme.fg("warning", `⏳ ${Math.round((Date.now() - wait.requestSentAt) / 1000)}s`));
}

function resetWaitState(ctx: ExtensionContext): void {
	clearWaitTimer();
	wait.requestSentAt = null;
	wait.firstTokenAt = null;
	wait.gotFirstToken = false;
	wait.ui = null;
	ctx.ui.setStatus(TTFT_STATUS_KEY, undefined);
}

// ---------------------------------------------------------------------------
// Extension registration
// ---------------------------------------------------------------------------

export default async function registerOpenAICompat(pi: ExtensionAPI): Promise<void> {
	const baseUrl = resolveBaseUrl();
	const apiKey = resolveApiKey();
	if (!baseUrl) return;

	let models: ProviderModelConfig[];
	try {
		models = (await discoverModels(baseUrl, apiKey ?? "no-key")).map(toProviderModel);
	} catch (err) {
		const msg = err instanceof Error ? err.message : String(err);
		console.error(`[pi-openai-compat] Failed to discover models from ${baseUrl}: ${msg}`);
		return;
	}
	if (models.length === 0) {
		console.error(`[pi-openai-compat] No models found at ${baseUrl}`);
		return;
	}

	pi.registerProvider(PROVIDER_NAME, {
		baseUrl,
		api: "openai-completions",
		apiKey: apiKey ?? "no-key",
		authHeader: true,
		models,
	});

	// Poll /user/info at load time for RPM/TPM limits.
	try {
		const infoUrl = baseUrl.replace(/\/v1\/?$/, "") + "/user/info";
		const resp = await fetch(infoUrl, {
			headers: { "Content-Type": "application/json", ...(apiKey ? { Authorization: `Bearer ${apiKey}` } : {}) },
		});
		if (resp.ok) {
			const data = await resp.json() as { user_info?: { tpm_limit?: number | null; rpm_limit?: number | null } };
			const ui = data.user_info;
			if (ui) {
				quota.tokenLimit = ui.tpm_limit ?? null;
				quota.rpmLimit = ui.rpm_limit ?? null;
			}
		}
	} catch (err) {
		console.error(`[pi-openai-compat] /user/info failed:`, err);
	}

	// RPM tracking — increment on request start so status bar is accurate.
	pi.on("before_provider_request", (_event, ctx) => {
		trackRequestStart();

		// Pre-turn budget check — warn if this request will likely hit the wall.
		if (quota.tokenLimit !== null) {
			const usage = ctx.getContextUsage();
			const ctxTokens = usage?.tokens ?? 0;
			const tpmUsed = getRollingTPM() + ctxTokens;
			const budgetPct = tpmUsed / quota.tokenLimit;
			if (budgetPct >= 0.9 && !lastBudgetNotified) {
				lastBudgetNotified = true;
				ctx.ui.notify(
					`⚠️ Pre-turn budget warning: context (${fmtTokens(ctxTokens)}) + current TPM usage (${fmtTokens(getRollingTPM())}) = ${fmtTokens(tpmUsed)}/${fmtTokens(quota.tokenLimit)}. This request may be cut off by the TPM limit. Consider compacting context or waiting.`,
					"warning"
				);
			}
		}
	});

	// 429 monitoring.
	pi.on("after_provider_response", (event, ctx) => {
		if (ctx.model?.provider !== PROVIDER_NAME) return;
		if (event.status === 429) {
			rateLimited.isLimited = true;
			logEvent(ctx, { ts: new Date().toISOString(), event: "429", details: { tpm: getRollingTPM(), rpm: getRollingRPM() } });
			lastNotifiedPct = 0;
			const retryRaw = event.headers["retry-after"];
			rateLimited.retryAfterSecs = retryRaw ? parseInt(retryRaw, 10) : null;
			if (rateLimited.retryAfterSecs !== null) {
				const resetDate = new Date(Date.now() + rateLimited.retryAfterSecs * 1000);
				rateLimited.resetAt = resetDate.toISOString().replace("T", " ").replace(/\.\d{3}Z$/, " UTC");
			}
			ctx.ui.notify(`⛔ Token quota exhausted.${rateLimited.resetAt ? ` Resets at ${fmtResetTime(rateLimited.resetAt)}.` : ""} Switch to a different model or wait for reset.`, "error");
		} else if (event.status === 200) {
			rateLimited.isLimited = false;
			rateLimited.resetAt = null;
			rateLimited.retryAfterSecs = null;
			lastNotifiedPct = null;
		}
		updateQuotaStatus(ctx);
	});

	// Session stats + token tracking.
	pi.on("message_end", async (event, ctx) => {
		if (ctx.model?.provider !== PROVIDER_NAME || event.message.role !== "assistant") return;
		const msg = event.message as unknown as Record<string, unknown>;
		const reason = msg.stop_reason ?? msg.stopReason ?? msg.finish_reason;
		const usage = ctx.getContextUsage();
		if (reason) session.finishReason = String(reason);
		if (reason && reason !== "end_turn" && reason !== "stop" && reason !== "stop_sequence") {
			logEvent(ctx, { ts: new Date().toISOString(), event: "finish", details: { reason, tokens: usage?.tokens } });
		}
		if (usage?.tokens) {
			session.contextUsed = usage.tokens;
			session.contextWindow = ctx.model?.contextWindow ?? null;
			trackTokens(usage.tokens);

			// Warn when context crosses thresholds — suggest compacting.
			const ctxPct = session.contextWindow && session.contextWindow > 0
				? usage.tokens / session.contextWindow
				: null;
			if (ctxPct !== null) {
				const crossed75 = ctxPct >= 0.75 && (lastContextNotifiedPct === null || lastContextNotifiedPct < 0.75);
				const crossed40 = ctxPct >= 0.40 && (lastContextNotifiedPct === null || lastContextNotifiedPct < 0.40);
				if (crossed75) {
					lastContextNotifiedPct = ctxPct;
					ctx.ui.notify(`⚠️ Context at ${Math.round(ctxPct * 100)}% — compact soon to avoid hitting the limit mid-turn.`, "warning");
				} else if (crossed40) {
					lastContextNotifiedPct = ctxPct;
					ctx.ui.notify(`💡 Context at ${Math.round(ctxPct * 100)}% — consider compacting to keep token costs low.`, "info");
				}
			}
		}
		updateSessionStatus(ctx);
	});

	// TTFT indicators.
	pi.on("turn_start", (_event, ctx) => {
		if (ctx.model?.provider !== PROVIDER_NAME) return;
		resetWaitState(ctx);
	});
	pi.on("before_provider_request", (_event, ctx) => {
		if (ctx.model?.provider !== PROVIDER_NAME) return;
		wait.requestSentAt = Date.now();
		wait.ui = ctx.ui;
		clearWaitTimer();
		tickWaitStatus();
		wait.timer = setInterval(tickWaitStatus, 1000);
	});
	pi.on("message_update", (event, ctx) => {
		if (ctx.model?.provider !== PROVIDER_NAME) return;
		if ((event.message as { role?: string }).role !== "assistant") return;
		if (wait.gotFirstToken) return;
		wait.gotFirstToken = true;
		wait.firstTokenAt = Date.now();
		clearWaitTimer();
		if (wait.requestSentAt && wait.firstTokenAt) {
			const ttft = ((wait.firstTokenAt - wait.requestSentAt) / 1000).toFixed(1);
			ctx.ui.setStatus(TTFT_STATUS_KEY, ctx.ui.theme.fg("dim", `⚡ ${ttft}s`));
		}
	});
	pi.on("turn_end", (_event, ctx) => {
		if (ctx.model?.provider !== PROVIDER_NAME) return;
		clearWaitTimer();
		if (!wait.gotFirstToken) ctx.ui.setStatus(TTFT_STATUS_KEY, undefined);
	});

	// Shared lifecycle.
	pi.on("model_select", (_event, ctx) => {
		updateQuotaStatus(ctx);
		updateSessionStatus(ctx);
		resetWaitState(ctx);
	});
	pi.on("session_start", async (_event, ctx) => {
		resetSessionStats();
		resetWaitState(ctx);
		lastContextNotifiedPct = null;
		updateQuotaStatus(ctx);
		updateSessionStatus(ctx);
	});
}
