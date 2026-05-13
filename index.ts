import type {
	ExtensionAPI,
	ExtensionContext,
	ProviderModelConfig,
	ThemeColor,
} from "@earendil-works/pi-coding-agent";

const PROVIDER_NAME = "openai-compat";

interface OpenAIModelEntry {
	id: string;
	owned_by?: string;
	// llama.cpp fields
	meta?: {
		n_ctx_train?: number;
	};
	// LiteLLM / OpenAI-compatible fields (may be present on /v1/models)
	max_tokens?: number;
	max_input_tokens?: number;
	context_window?: number;
	// Merged in from LiteLLM /model/info
	supportsVision?: boolean;
}

// LiteLLM-specific /model/info response shape
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

// Known context windows for common models — fallback when the API is silent.
const KNOWN_CONTEXT_WINDOWS: Array<[RegExp, number]> = [
	[/qwen3/i, 131072],
	[/qwq-32b/i, 131072],
	[/deepseek-r1/i, 131072],
];

// Known reasoning/thinking model ID patterns (case-insensitive).
// LiteLLM's supports_reasoning field is unreliable (often null), so we detect by name.
const REASONING_MODEL_PATTERNS = [
	/qwen3/i,
	/qwq/i,
	/deepseek-r1/i,
	/deepseek-reasoner/i,
];

// LiteLLM supports stream_options.include_usage; llama.cpp does not.
// Default: enabled. Set OPENAI_COMPAT_NO_STREAMING_USAGE=1 for llama.cpp backends.
const STREAMING_USAGE_ENABLED = process.env.OPENAI_COMPAT_NO_STREAMING_USAGE !== "1";

function resolveBaseUrl(): string | undefined {
	return (
		process.env.OPENAI_COMPAT_BASE_URL ??
		process.env.OPENAI_BASE_URL
	);
}

function resolveApiKey(): string | undefined {
	return (
		process.env.OPENAI_COMPAT_API_KEY ??
		process.env.OPENAI_API_KEY
	);
}

async function fetchJson<T>(url: string, headers: Record<string, string>): Promise<T | null> {
	try {
		const response = await fetch(url, {
			signal: AbortSignal.timeout(5000),
			headers,
		});
		if (!response.ok) return null;
		return (await response.json()) as T;
	} catch {
		return null;
	}
}

async function discoverModels(baseUrl: string, apiKey?: string): Promise<OpenAIModelEntry[]> {
	const base = baseUrl.replace(/\/+$/, "").replace(/\/v1$/, "");
	const headers: Record<string, string> = { Accept: "application/json" };
	if (apiKey) {
		headers["Authorization"] = `Bearer ${apiKey}`;
	}

	// Primary: standard OpenAI /v1/models
	const modelsBody = await fetchJson<OpenAIModelsResponse>(`${base}/v1/models`, headers);
	if (!modelsBody) {
		throw new Error("Failed to fetch /v1/models");
	}

	// llama.cpp returns both .data (OpenAI standard) and .models (legacy)
	const entries = (modelsBody.data ?? modelsBody.models ?? [])
		.filter((m) => m.id && m.id.length > 0);

	// Secondary: LiteLLM /model/info for richer metadata (context window, vision).
	// Non-fatal — plain OpenAI/llama.cpp endpoints without this route are unaffected.
	const infoBody = await fetchJson<LiteLLMModelInfoResponse>(`${base}/model/info`, headers);
	if (infoBody?.data) {
		const infoByName = new Map<string, LiteLLMModelInfo>();
		for (const item of infoBody.data) {
			infoByName.set(item.model_name, item);
		}
		for (const entry of entries) {
			const info = infoByName.get(entry.id);
			if (!info?.model_info) continue;
			const mi = info.model_info;
			// max_tokens on model_info is the context window (LiteLLM convention)
			if (mi.max_tokens != null && mi.max_tokens > 0) {
				entry.context_window = mi.max_tokens;
			}
			if (mi.max_input_tokens != null && mi.max_input_tokens > 0) {
				entry.max_input_tokens = mi.max_input_tokens;
			}
			if (mi.supports_vision != null) {
				entry.supportsVision = mi.supports_vision;
			}
		}
	}

	return entries;
}

/**
 * Derive the model ID to register with pi from a raw ramalama/llama.cpp entry.
 *
 * ramalama prefixes model IDs with a registry namespace, e.g.
 * "library/qwen2.5-coder". That embedded '/' creates a 3-segment pi model
 * reference ("openai-compat/library/qwen2.5-coder") which breaks the
 * --models glob "openai-compat/*" (minimatch's * does not cross '/').
 *
 * ramalama also accepts the bare basename in chat/completions requests, so
 * we register just the basename as the id and use the full original as the
 * display name.
 */
function resolveModelId(entry: OpenAIModelEntry): string {
	const slashIdx = entry.id.lastIndexOf("/");
	return slashIdx !== -1 ? entry.id.slice(slashIdx + 1) : entry.id;
}

function isReasoningModel(modelId: string): boolean {
	return REASONING_MODEL_PATTERNS.some((pattern) => pattern.test(modelId));
}

function resolveContextWindow(entry: OpenAIModelEntry): number {
	// Prefer API-reported values (first non-zero wins)
	const fromApi =
		entry.context_window ||
		entry.max_input_tokens ||
		entry.max_tokens ||
		entry.meta?.n_ctx_train;
	if (fromApi && fromApi > 0) return fromApi;

	// Fall back to known-model table
	for (const [pattern, size] of KNOWN_CONTEXT_WINDOWS) {
		if (pattern.test(entry.id)) return size;
	}

	return 32768;
}

function toProviderModel(entry: OpenAIModelEntry): ProviderModelConfig {
	const contextWindow = resolveContextWindow(entry);
	const reasoning = isReasoningModel(entry.id);
	// Reasoning models need more output headroom for thinking traces.
	const maxTokens = reasoning
		? Math.min(16384, contextWindow)
		: Math.min(4096, contextWindow);

	const input: ProviderModelConfig["input"] = entry.supportsVision
		? ["text", "image"]
		: ["text"];

	return {
		id: resolveModelId(entry),  // strips ramalama namespace prefix if present
		name: entry.id,             // full original ID shown in model picker
		reasoning,
		input,
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow,
		maxTokens,
		compat: {
			// llama.cpp uses max_tokens, not max_completion_tokens
			maxTokensField: "max_tokens",
			// llama.cpp rejects the "developer" system-prompt role
			supportsDeveloperRole: false,
			// llama.cpp does not support reasoning_effort
			supportsReasoningEffort: false,
			// LiteLLM supports stream_options.include_usage (enables context tracking).
			// Set OPENAI_COMPAT_NO_STREAMING_USAGE=1 to disable for llama.cpp backends.
			supportsUsageInStreaming: STREAMING_USAGE_ENABLED,
			// Qwen3 on vLLM requires chat_template_kwargs: {enable_thinking: true}
			// to activate the reasoning chain. Non-Qwen models leave this unset.
			...(reasoning ? { thinkingFormat: "qwen-chat-template" } : {}),
		},
	};
}

// ---------------------------------------------------------------------------
// Footer state — quota + session stats shown in Pi status bar
// ---------------------------------------------------------------------------

interface QuotaState {
	/** True when a 429 has been received and not yet cleared by a 200. */
	isLimited: boolean;
	/** Reset timestamp from Reset_at header (e.g. "2026-05-13 00:57:30 UTC"). */
	resetAt: string | null;
	/** Retry-After seconds from 429 response. */
	retryAfterSecs: number | null;
	/** Remaining user-level tokens from X-Ratelimit-User-Remaining-Tokens. */
	remaining: number | null;
	/** User-level token limit from X-Ratelimit-User-Limit-Tokens. */
	limit: number | null;
}

interface SessionStats {
	/** Finish reason from last completed assistant message. */
	finishReason: string | null;
	/** Prompt tokens used in last turn (from usage). */
	contextUsed: number | null;
	/** Model context window size. */
	contextWindow: number | null;
}

const quota: QuotaState = {
	isLimited: false,
	resetAt: null,
	retryAfterSecs: null,
	remaining: null,
	limit: null,
};

const session: SessionStats = {
	finishReason: null,
	contextUsed: null,
	contextWindow: null,
};

const QUOTA_STATUS_KEY  = "openai-compat-quota";
const SESSION_STATUS_KEY = "openai-compat-session";
const TTFT_STATUS_KEY    = "openai-compat-ttft";

// Finish reason display: icon + label + colour
const FINISH_REASON_MAP: Record<string, { icon: string; label: string; color: string }> = {
	end_turn:      { icon: "⏹",  label: "done",       color: "dim" },
	stop:          { icon: "⏹",  label: "done",       color: "dim" },
	stop_sequence: { icon: "⏹",  label: "stop_seq",   color: "dim" },
	max_tokens:    { icon: "✂",  label: "max_tokens", color: "warning" },
	length:        { icon: "✂",  label: "max_tokens", color: "warning" },
	tool_use:      { icon: "🔧", label: "tool",       color: "dim" },
	tool_calls:    { icon: "🔧", label: "tool",       color: "dim" },
	error:         { icon: "⚠",  label: "error",      color: "error" },
};

function fmtTokens(n: number): string {
	if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
	if (n >= 1_000) return `${Math.round(n / 1_000)}K`;
	return String(n);
}

function fmtResetTime(resetAt: string): string {
	// "2026-05-13 00:57:30 UTC" → "00:57 UTC"
	const m = resetAt.match(/(\d{2}:\d{2}):\d{2} (UTC)/i);
	return m ? `${m[1]} ${m[2]}` : resetAt;
}

function updateQuotaStatus(ctx: ExtensionContext): void {
	if (ctx.model?.provider !== PROVIDER_NAME) {
		ctx.ui.setStatus(QUOTA_STATUS_KEY, undefined);
		return;
	}

	if (quota.isLimited) {
		const reset = quota.resetAt
			? `reset ${fmtResetTime(quota.resetAt)}`
			: quota.retryAfterSecs
				? `retry in ${quota.retryAfterSecs}s`
				: "rate limited";
		ctx.ui.setStatus(QUOTA_STATUS_KEY, ctx.ui.theme.fg("error", `⛔ ${reset}`));
		return;
	}

	if (quota.remaining !== null && quota.limit !== null && quota.limit > 0) {
		const pct = quota.remaining / quota.limit;
		const color = (pct < 0.10 ? "error" : pct < 0.25 ? "warning" : "dim") as ThemeColor;
		const label = `tkn: ${fmtTokens(quota.remaining)}/${fmtTokens(quota.limit)}`;
		ctx.ui.setStatus(QUOTA_STATUS_KEY, ctx.ui.theme.fg(color, label));
		return;
	}

	ctx.ui.setStatus(QUOTA_STATUS_KEY, undefined);
}

function updateSessionStatus(ctx: ExtensionContext): void {
	if (ctx.model?.provider !== PROVIDER_NAME) {
		ctx.ui.setStatus(SESSION_STATUS_KEY, undefined);
		return;
	}

	const parts: string[] = [];

	// Context usage: e.g. "42K/131K ctx"
	if (session.contextUsed !== null && session.contextWindow !== null && session.contextWindow > 0) {
		const pct = session.contextUsed / session.contextWindow;
		const color = pct > 0.85 ? "error" : pct > 0.65 ? "warning" : "dim";
		parts.push(ctx.ui.theme.fg(color,
			`${fmtTokens(session.contextUsed)}/${fmtTokens(session.contextWindow)} ctx`
		));
	}

	// Finish reason: e.g. "✂ max_tokens" or "⏹ done"
	if (session.finishReason) {
		const def = FINISH_REASON_MAP[session.finishReason] ?? {
			icon: "?", label: session.finishReason, color: "dim",
		};
		parts.push(ctx.ui.theme.fg(def.color as ThemeColor, `${def.icon} ${def.label}`));
	}

	if (parts.length > 0) {
		ctx.ui.setStatus(SESSION_STATUS_KEY,
			parts.join(ctx.ui.theme.fg("dim", "  "))
		);
	} else {
		ctx.ui.setStatus(SESSION_STATUS_KEY, undefined);
	}
}

function resetSessionStats(): void {
	session.finishReason = null;
	session.contextUsed = null;
	session.contextWindow = null;
}

// ---------------------------------------------------------------------------
// TTFT (time-to-first-token) waiting indicator
//
// With LiteMaaS + Qwen3, the model reasons inside <think>...</think> tags
// that LiteLLM strips mid-stream. From Pi's perspective the connection is
// open but silent — no SSE chunks arrive until thinking is complete.
// This produces the "model stalled" appearance with no observable signal.
//
// Fix: start a setInterval on before_provider_request that updates a status
// slot every second with elapsed time. First message_update stops the timer
// and shows the TTFT. The elapsed counter proves Pi is alive and waiting;
// the TTFT metric is a proxy for how long the model reasoned.
// ---------------------------------------------------------------------------

interface WaitState {
	/** Timestamp when before_provider_request fired for this turn. */
	requestSentAt: number | null;
	/** Timestamp when the first message_update token arrived. */
	firstTokenAt: number | null;
	/** True once the first token has been received this turn. */
	gotFirstToken: boolean;
	/** Active setInterval handle. */
	timer: ReturnType<typeof setInterval> | null;
	/** Captured ctx.ui reference for use inside the interval callback. */
	ui: ExtensionContext["ui"] | null;
}

const wait: WaitState = {
	requestSentAt: null,
	firstTokenAt: null,
	gotFirstToken: false,
	timer: null,
	ui: null,
};

function clearWaitTimer(): void {
	if (wait.timer !== null) {
		clearInterval(wait.timer);
		wait.timer = null;
	}
}

function tickWaitStatus(): void {
	if (!wait.ui || !wait.requestSentAt || wait.gotFirstToken) return;
	const elapsed = Math.round((Date.now() - wait.requestSentAt) / 1000);
	wait.ui.setStatus(TTFT_STATUS_KEY, wait.ui.theme.fg("warning", `⏳ ${elapsed}s`));
}

function resetWaitState(ctx: ExtensionContext): void {
	clearWaitTimer();
	wait.requestSentAt = null;
	wait.firstTokenAt  = null;
	wait.gotFirstToken = false;
	wait.ui            = null;
	ctx.ui.setStatus(TTFT_STATUS_KEY, undefined);
}

export default async function registerOpenAICompat(pi: ExtensionAPI): Promise<void> {
	const baseUrl = resolveBaseUrl();
	const apiKey = resolveApiKey();

	if (!baseUrl) {
		// No endpoint configured — silently skip registration.
		return;
	}

	let models: ProviderModelConfig[];
	try {
		const entries = await discoverModels(baseUrl, apiKey ?? "no-key");
		models = entries.map(toProviderModel);
	} catch (err) {
		const msg = err instanceof Error ? err.message : String(err);
		console.error(`[pi-openai-compat] Failed to discover models from ${baseUrl}: ${msg}`);
		console.error("[pi-openai-compat] Provider will not be registered.");
		return;
	}

	if (models.length === 0) {
		console.error(`[pi-openai-compat] No models found at ${baseUrl}. Provider will not be registered.`);
		return;
	}

	pi.registerProvider(PROVIDER_NAME, {
		baseUrl,
		api: "openai-completions",
		apiKey: apiKey ?? "no-key",
		authHeader: true,
		models,
	});

	// -------------------------------------------------------------------------
	// Proactive quota poll — fetch /user/info at session start
	//
	// The rate-limit headers on 200 responses update the quota slot reactively,
	// but only after the first request. LiteLLM's /user/info endpoint returns
	// the current budget state — poll it on session_start so the status slot
	// shows remaining tokens before any request is made.
	// -------------------------------------------------------------------------

	pi.on("session_start", async (_event, ctx) => {
		const infoUrl = baseUrl.replace(/\/v1\/?$/, "") + "/user/info";
		try {
			const headers: Record<string, string> = {
				"Content-Type": "application/json",
			};
			if (apiKey) headers["Authorization"] = `Bearer ${apiKey}`;

			const resp = await fetch(infoUrl, { headers });
			if (!resp.ok) return;

			const data = await resp.json() as {
				keys?: Array<{
					token_budget_duration?: string | null;
					max_budget?: number | null;
					spend?: number | null;
					tpm_limit?: number | null;
					rpm_limit?: number | null;
				}>;
				teams?: Array<{
					tpm_limit?: number | null;
					max_budget?: number | null;
					spend?: number | null;
				}>;
			};

			// Prefer team-level TPM limit (reflects the per-user token quota)
			// Fall back to key-level limits if present
			const team = data.teams?.[0];
			const key = data.keys?.[0];

			// LiteLLM tracks spend in dollars; the 429 error says "Limit type: tokens"
			// meaning the limit tracked here is token-based, not dollar-based.
			// The X-Ratelimit-User-Limit-Tokens header (5000000) matches the user tpm_limit.
			const tpmLimit = team?.tpm_limit ?? key?.tpm_limit ?? null;
			if (tpmLimit !== null) {
				quota.limit = tpmLimit;
			}

			updateQuotaStatus(ctx);
		} catch {
			// Non-fatal — quota display degrades gracefully
		}
	});

	// -------------------------------------------------------------------------
	// Quota monitoring — rate limit headers → quota status slot
	// -------------------------------------------------------------------------

	pi.on("after_provider_response", (event, ctx) => {
		if (ctx.model?.provider !== PROVIDER_NAME) return;

		const h = event.headers;

		if (event.status === 429) {
			quota.isLimited = true;
			quota.resetAt = h["reset_at"] ?? null;
			const retryRaw = h["retry-after"];
			quota.retryAfterSecs = retryRaw ? parseInt(retryRaw, 10) : null;
			quota.remaining = 0;
		} else if (event.status === 200) {
			quota.isLimited = false;
			quota.resetAt = null;
			quota.retryAfterSecs = null;
			// User-level limits are what trigger 429 — prefer those over pool limits
			const userRemaining = h["x-ratelimit-user-remaining-tokens"];
			const userLimit     = h["x-ratelimit-user-limit-tokens"];
			const poolRemaining = h["x-ratelimit-remaining-tokens"];
			const poolLimit     = h["x-ratelimit-limit-tokens"];
			quota.remaining = userRemaining !== undefined
				? parseInt(userRemaining, 10)
				: poolRemaining !== undefined ? parseInt(poolRemaining, 10) : null;
			quota.limit = userLimit !== undefined
				? parseInt(userLimit, 10)
				: poolLimit !== undefined ? parseInt(poolLimit, 10) : null;
		}

		updateQuotaStatus(ctx);
	});

	// -------------------------------------------------------------------------
	// Session stats — finish reason + context usage → session status slot
	// -------------------------------------------------------------------------

	pi.on("message_end", async (event, ctx) => {
		if (ctx.model?.provider !== PROVIDER_NAME) return;
		if (event.message.role !== "assistant") return;

		// eslint-disable-next-line @typescript-eslint/no-explicit-any
		const msg = event.message as any;

		// Finish reason — Pi maps OpenAI finish_reason to Anthropic stop_reason internally
		const reason = msg.stop_reason ?? msg.stopReason ?? msg.finish_reason ?? null;
		if (reason) session.finishReason = String(reason);

		// Context usage — populated by Pi when supportsUsageInStreaming is true
		const usage = ctx.getContextUsage();
		if (usage?.tokens) {
			session.contextUsed = usage.tokens;
			session.contextWindow = ctx.model?.contextWindow ?? null;
		}

		updateSessionStatus(ctx);
	});

	// -------------------------------------------------------------------------
	// TTFT waiting indicator — turn_start / before_provider_request / message_update
	// -------------------------------------------------------------------------

	// Reset per-turn state before each LLM turn.
	pi.on("turn_start", (_event, ctx) => {
		if (ctx.model?.provider !== PROVIDER_NAME) return;
		resetWaitState(ctx);
	});

	// Request dispatched — start the elapsed-time ticker.
	pi.on("before_provider_request", (_event, ctx) => {
		if (ctx.model?.provider !== PROVIDER_NAME) return;
		wait.requestSentAt = Date.now();
		wait.ui            = ctx.ui;
		clearWaitTimer();
		tickWaitStatus();  // show immediately rather than waiting for first tick
		wait.timer = setInterval(tickWaitStatus, 1000);
	});

	// First token arrived — stop ticker, show TTFT, then dim after 5s.
	pi.on("message_update", (event, ctx) => {
		if (ctx.model?.provider !== PROVIDER_NAME) return;
		if ((event.message as { role?: string }).role !== "assistant") return;
		if (wait.gotFirstToken) return;

		wait.gotFirstToken = true;
		wait.firstTokenAt  = Date.now();
		clearWaitTimer();

		if (wait.requestSentAt && wait.firstTokenAt) {
			const ttft = ((wait.firstTokenAt - wait.requestSentAt) / 1000).toFixed(1);
			ctx.ui.setStatus(TTFT_STATUS_KEY, ctx.ui.theme.fg("dim", `⚡ ${ttft}s`));
		}
	});

	// Safety net: if the turn ends without any tokens (abort / error), clear.
	pi.on("turn_end", (_event, ctx) => {
		if (ctx.model?.provider !== PROVIDER_NAME) return;
		clearWaitTimer();
		if (!wait.gotFirstToken) {
			ctx.ui.setStatus(TTFT_STATUS_KEY, undefined);
		}
	});

	// -------------------------------------------------------------------------
	// Shared lifecycle hooks
	// -------------------------------------------------------------------------

	pi.on("model_select", (_event, ctx) => {
		updateQuotaStatus(ctx);
		updateSessionStatus(ctx);
		resetWaitState(ctx);
	});

	pi.on("session_start", async (_event, ctx) => {
		resetSessionStats();
		resetWaitState(ctx);
		updateQuotaStatus(ctx);
		updateSessionStatus(ctx);
	});
}
