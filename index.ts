import type {
	ExtensionAPI,
	ProviderModelConfig,
} from "@mariozechner/pi-coding-agent";

const PROVIDER_NAME = "openai-compat";

interface OpenAIModelEntry {
	id: string;
	owned_by?: string;
	meta?: {
		n_ctx_train?: number;
	};
}

interface OpenAIModelsResponse {
	data?: OpenAIModelEntry[];
	models?: OpenAIModelEntry[];
}

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

async function discoverModels(baseUrl: string): Promise<OpenAIModelEntry[]> {
	const modelsUrl = baseUrl.replace(/\/+$/, "").replace(/\/v1$/, "") + "/v1/models";

	const response = await fetch(modelsUrl, {
		signal: AbortSignal.timeout(5000),
		headers: {
			Accept: "application/json",
		},
	});

	if (!response.ok) {
		throw new Error(`${response.status} ${response.statusText}`);
	}

	const body = (await response.json()) as OpenAIModelsResponse;

	// llama.cpp returns both .data (OpenAI standard) and .models (legacy)
	const entries = body.data ?? body.models ?? [];
	return entries.filter((m) => m.id && m.id.length > 0);
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

function toProviderModel(entry: OpenAIModelEntry): ProviderModelConfig {
	const contextWindow = entry.meta?.n_ctx_train ?? 32768;

	return {
		id: resolveModelId(entry),   // "qwen2.5-coder" — API accepts this
		name: entry.id,              // "library/qwen2.5-coder" — shown in /model
		reasoning: false,
		input: ["text"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow,
		maxTokens: Math.min(4096, contextWindow),
		compat: {
			// llama.cpp uses max_tokens, not max_completion_tokens
			maxTokensField: "max_tokens",
			// llama.cpp rejects the "developer" system-prompt role
			supportsDeveloperRole: false,
			// llama.cpp does not support reasoning_effort
			supportsReasoningEffort: false,
			// llama.cpp does not support stream_options.include_usage
			supportsUsageInStreaming: false,
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
		const entries = await discoverModels(baseUrl);
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
