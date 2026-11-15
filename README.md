# Qubax AI for Hermes Agent

Use [Qubax AI](https://qubax.ai) inside [Hermes Agent](https://hermes-agent.nousresearch.com) with one key. You get 400+ models for chat, images, video and voice, often far below list price, and you can pay with crypto.

This repo has two plugins:

| Plugin | What it adds |
| --- | --- |
| `provider` | Qubax as a **chat model provider** (200+ text models) |
| `media` | Qubax for **image generation**, **video generation**, **text-to-speech** and **speech-to-text** |

## Install

```bash
hermes plugins install QubaxAi/hermes-qubax/provider
hermes plugins install QubaxAi/hermes-qubax/media
```

1. Get an API key at **https://qubax.ai/app/api-keys** (no card needed).
2. Add it once: `echo 'QUBAX_API_KEY=qbx_live_...' >> ~/.hermes/.env`
3. Chat: run `hermes model` and pick **Qubax AI**, or `hermes --provider qubax -m glm-5.3-flash`.
4. Media: run `hermes tools` and pick **Qubax AI** for Image, Video, Voice and Transcription, or add this to `~/.hermes/config.yaml`:

```yaml
image_gen: {provider: qubax}
video_gen: {provider: qubax}
tts:       {provider: qubax}
stt:       {provider: qubax}
```

## Chat models

- **Every Qubax text model (200+) in the picker.** Featured models come first, cheapest first: GLM 5.3 Flash, Qwen 3.8, DeepSeek V4.1 Flash, Claude Haiku 5.5, GPT-6.1 Sol, Claude Opus 5.5 and more. All other models follow. You can also type any id directly, e.g. `/model kimi-k3`.
- **Cheap default for background work.** Compression, titles and summaries use the cheapest featured model.
- **Correct context windows** and **reasoning effort** capped at what each model allows, so a request never costs more than you asked for.
- **`/usage` shows your Qubax balance.**

## Media models

| Hermes feature | Qubax models | Default |
| --- | --- | --- |
| Image generation and editing | 50+ (FLUX, Seedream, Nano Banana, GPT Image, Qwen Image…) | cheapest featured image model |
| Video generation (text and image to video) | 100+ (Seedance, Kling, Wan, Veo, Grok Imagine…) | cheapest featured text-to-video model |
| Text-to-speech | ElevenLabs, MiniMax, Kokoro, OpenAI TTS and more | cheapest featured voice |
| Speech-to-text | Whisper, ElevenLabs Scribe, xAI and more | cheapest featured transcriber |

- Lists are read live from Qubax, so new models show up without a plugin update.
- When you ask to edit an image or animate a picture, the plugin switches to a model that supports it.
- To pin a model, set `model:` under the section, e.g. `image_gen: {provider: qubax, model: flux-2-pro}`.
- Failed video jobs are not charged.

Browse every model and price at https://qubax.ai/models.

## Settings

| Variable | Purpose |
| --- | --- |
| `QUBAX_API_KEY` | Your Qubax key (required) |
| `QUBAX_BASE_URL` | Override the endpoint (default `https://api.qubax.ai/v1`) |

## Privacy and security

- The plugins talk only to `api.qubax.ai`. The model list comes from the public catalog; your key is sent only on chat, generation and balance requests, and is never logged.
- No telemetry, no third-party services, no extra Python packages.

## Links

- Website: https://qubax.ai
- Models and prices: https://qubax.ai/models
- Support: https://t.me/QubaxAiChat

MIT licensed.
