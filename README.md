# Qubax AI for Hermes Agent

Adds **Qubax AI** as a model provider in [Hermes Agent](https://hermes-agent.nousresearch.com).
Qubax gives you 400+ AI models through one OpenAI-compatible API, often far below list price, and you can pay with crypto.

## Install

```bash
hermes plugins install QubaxAi/hermes-qubax
```

1. Get an API key at **https://qubax.ai/app/api-keys** (no card needed).
2. Run `hermes model`, pick **Qubax AI**, paste your key, then choose a model.

Or set it directly:

```bash
echo 'QUBAX_API_KEY=qbx_live_...' >> ~/.hermes/.env
hermes --provider qubax -m glm-5.3-flash
```

## What you get

- **Every Qubax text model in the picker (200+).** Featured models come first, cheapest first: GLM 5.3 Flash, Qwen 3.8, DeepSeek V4.1 Flash, Claude Haiku 5.5, GPT-6.1 Sol, Claude Opus 5.5 and more. All other models follow, also cheapest first. The list is read live from Qubax, so new models show up without a plugin update.
- **Cheap default for background work.** Compression, titles and summaries use the cheapest featured text model, not your main model.
- **Correct context windows** for every Qubax model.
- **Reasoning effort** is sent only to models that support it and is capped at the levels each model allows, so a request never costs more than you asked for.
- **`/usage` shows your Qubax balance.**

You can also type any model id directly, e.g. `/model kimi-k3`. Browse prices at https://qubax.ai/models.

## Settings

| Variable | Purpose |
| --- | --- |
| `QUBAX_API_KEY` | Your Qubax key (required) |
| `QUBAX_BASE_URL` | Override the endpoint (default `https://api.qubax.ai/v1`) |

## Privacy

The plugin talks only to `api.qubax.ai`. The model list comes from the public catalog; your key is sent only on chat and balance requests.

## Links

- Website: https://qubax.ai
- Models and prices: https://qubax.ai/models
- Support: https://t.me/QubaxAiChat

MIT licensed.
