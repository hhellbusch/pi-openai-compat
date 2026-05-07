import type {
	ExtensionAPI,
	ProviderModelConfig,
} from "@mariozechner/pi-coding-agent";

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
			// llama.cpp does not support stream_options.include_usage
			supportsUsageInStreaming: false,
			// Qwen3 on vLLM requires chat_template_kwargs: {enable_thinking: true}
			// to activate the reasoning chain. Non-Qwen models leave this unset.
			...(reasoning ? { thinkingFormat: "qwen-chat-template" } : {}),
		},
	};
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
}
