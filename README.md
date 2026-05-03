# pi-openai-compat

Pi extension for private OpenAI-compatible LLM endpoints.

Discovers models automatically from any endpoint that implements the
[OpenAI `/v1/models`](https://platform.openai.com/docs/api-reference/models/list) API.
Works with RamaLama, vLLM, llm-d, TGI, Ollama, and any other
OpenAI-compatible inference server.

## Installation

```bash
pi install git:github.com/hhellbusch/pi-openai-compat
```

## Configuration

Set environment variables before starting Pi:

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_COMPAT_BASE_URL` | Yes* | Base URL of the endpoint (e.g., `https://llm.lan:8443/v1`) |
| `OPENAI_COMPAT_API_KEY` | No | API key (sent as `Authorization: Bearer`). Defaults to `no-key`. |

*Falls back to `OPENAI_BASE_URL` / `OPENAI_API_KEY` if the `_COMPAT_` variants aren't set.

## How it works

At startup, the extension:

1. Reads the base URL from environment
2. Fetches `/v1/models` to discover available models
3. Registers each model under the `openai-compat` provider

Models appear in the Pi picker as `(openai-compat) model-id`.

If the endpoint is unreachable, the extension logs a warning and registers
nothing — Pi continues normally with other providers.

## Provider details

- **API type:** `openai-completions` (Pi's built-in OpenAI streaming)
- **Default context window:** 32768 tokens (auto-detected from endpoint when available)
- **Default max output:** 4096 tokens
- **Cost:** $0 (private inference)
