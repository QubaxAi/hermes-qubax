"""Qubax AI media backends for Hermes Agent: image, video, text-to-speech and speech-to-text.

One Qubax key (``QUBAX_API_KEY``) powers all four. Model lists come live from the public
catalog at ``https://api.qubax.ai/v1/models`` (featured first, then cheapest first), so new
models appear without a plugin update. The plugin talks only to api.qubax.ai; the key is
sent only on generation requests and is never logged.

Select a backend in config.yaml (or with ``hermes tools``)::

    image_gen: {provider: qubax}      # optional: model: z-image-turbo
    video_gen: {provider: qubax}      # optional: model: seedance-2-fast-t2v
    tts:       {provider: qubax}      # optional: model: tts-1, voice: ...
    stt:       {provider: qubax}      # optional: model: whisper-1
"""

from __future__ import annotations

import base64
import json
import logging
import mimetypes
import os
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any, Optional

from agent.image_gen_provider import ImageGenProvider, resolve_aspect_ratio
from agent.image_gen_provider import error_response as image_error
from agent.image_gen_provider import save_b64_image, save_url_image
from agent.image_gen_provider import success_response as image_success
from agent.transcription_provider import TranscriptionProvider
from agent.tts_provider import TTSProvider
from agent.video_gen_provider import VideoGenProvider, save_url_video
from agent.video_gen_provider import error_response as video_error
from agent.video_gen_provider import success_response as video_success

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.qubax.ai/v1"
KEY_ENV = "QUBAX_API_KEY"
SIGNUP_URL = "https://qubax.ai/app/api-keys"
_CATALOG_TTL_S = 15 * 60
_VIDEO_POLL_S = 5.0
_VIDEO_DEADLINE_S = 15 * 60

_lock = threading.Lock()
_catalog: dict[str, dict[str, Any]] | None = None
_catalog_at = 0.0


# ── shared helpers ─────────────────────────────────────────────────────────

def _secret(name: str) -> str:
    try:
        from agent.secret_scope import get_secret

        return (get_secret(name, "") or "").strip()
    except Exception:
        return (os.environ.get(name) or "").strip()


def _api_key() -> str:
    return _secret(KEY_ENV)


def _base_url() -> str:
    return (_secret("QUBAX_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")


def _user_agent() -> str:
    try:
        from hermes_cli.version_info import hermes_user_agent

        return hermes_user_agent("hermes-qubax-media")
    except Exception:
        return "hermes-qubax-media/1.0"


def _headers(json_body: bool = True) -> dict[str, str]:
    h = {"Authorization": f"Bearer {_api_key()}", "User-Agent": _user_agent(), "Accept": "application/json"}
    if json_body:
        h["Content-Type"] = "application/json"
    return h


def _error_text(resp: Any) -> str:
    """Short, key-free error text from a Qubax error response."""
    try:
        data = resp.json()
        msg = (data.get("error") or {}).get("message") if isinstance(data, dict) else None
        if msg:
            return f"HTTP {resp.status_code}: {str(msg)[:300]}"
    except Exception:
        pass
    return f"HTTP {resp.status_code}: {resp.text[:300]}"


def _load_catalog(*, timeout: float = 8.0) -> dict[str, dict[str, Any]]:
    """Public catalog keyed by id, cached 15 min; {} on failure (never raises)."""
    global _catalog, _catalog_at
    with _lock:
        if _catalog is not None and time.monotonic() - _catalog_at < _CATALOG_TTL_S:
            return _catalog
    try:
        req = urllib.request.Request(_base_url() + "/models",
                                     headers={"Accept": "application/json", "User-Agent": _user_agent()})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            items = json.loads(resp.read().decode()).get("data") or []
        cat = {m["id"]: m for m in items if isinstance(m, dict) and m.get("id")}
    except Exception as exc:
        logger.debug("qubax-media: catalog fetch failed: %s", exc)
        with _lock:
            return _catalog or {}
    with _lock:
        _catalog, _catalog_at = cat, time.monotonic()
    return cat


def _unit_price(m: dict[str, Any]) -> float:
    try:
        return float((m.get("pricing_summary") or {}).get("sell_per_unit") or 0)
    except (TypeError, ValueError):
        return 0.0


def _ready(m: dict[str, Any]) -> bool:
    return m.get("is_ready", True) is not False


def _sorted(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Featured first; inside each group cheapest first ($0 rows last), then most used."""
    def key(m: dict[str, Any]) -> tuple:
        p = _unit_price(m)
        return (0 if m.get("featured") else 1, p <= 0, p, -(m.get("requests_24h") or 0))
    return sorted(rows, key=key)


def _price_label(m: dict[str, Any]) -> str:
    p = _unit_price(m)
    unit = m.get("media_unit") or "unit"
    return f"${p:.4f}/{unit}" if p > 0 else ""


def _row(m: dict[str, Any]) -> dict[str, Any]:
    row = {"id": m["id"], "display": m.get("name") or m["id"],
           "strengths": (m.get("description") or "")[:90]}
    if price := _price_label(m):
        row["price"] = price
    if m.get("featured"):
        row["speed"] = "featured"
    return row


def _models(kind: str) -> list[dict[str, Any]]:
    cat = _load_catalog()
    outs = {"image": ["image"], "video": ["video"], "tts": ["audio"]}

    def match(m: dict[str, Any]) -> bool:
        if not _ready(m):
            return False
        if kind == "stt":
            ins = m.get("input_modalities") or []
            return m.get("output_modalities") == ["text"] and "audio" in ins and "text" not in ins
        return (m.get("output_modalities") or []) == outs[kind]

    return _sorted([m for m in cat.values() if match(m)])


def _first(rows: list[dict[str, Any]], pred=lambda m: True) -> Optional[str]:
    for m in rows:
        if _unit_price(m) > 0 and pred(m):
            return m["id"]
    return None


def _setup_schema(name: str, tag: str) -> dict[str, Any]:
    return {"name": name, "badge": "paid", "tag": tag,
            "env_vars": [{"key": KEY_ENV, "prompt": "Qubax API key", "url": SIGNUP_URL}]}


def _config(section: str) -> dict[str, Any]:
    try:
        from hermes_cli.config import load_config

        sec = (load_config() or {}).get(section)
        return sec if isinstance(sec, dict) else {}
    except Exception:
        return {}


def _configured_model(section: str) -> Optional[str]:
    """``<section>.qubax.model`` then ``<section>.model`` (only when provider is qubax)."""
    sec = _config(section)
    raw_sub = sec.get("qubax")
    sub: dict[str, Any] = raw_sub if isinstance(raw_sub, dict) else {}
    model = sub.get("model") or (sec.get("model") if str(sec.get("provider") or "").lower() == "qubax" else None)
    return model.strip() if isinstance(model, str) and model.strip() else None


def _to_data_url(src: str) -> str:
    """Local file → data: URL; URLs and data: URLs pass through unchanged."""
    s = (src or "").strip()
    if not s or s.lower().startswith(("http://", "https://", "data:")):
        return s
    p = Path(os.path.expanduser(s))
    if not p.is_file():
        raise ValueError(f"Source image not found: {s}")
    if p.stat().st_size > 10 * 1024 * 1024:
        raise ValueError("Source image is larger than 10 MB.")
    mime = mimetypes.guess_type(p.name)[0] or "image/png"
    return f"data:{mime};base64," + base64.b64encode(p.read_bytes()).decode("ascii")


# ── image ──────────────────────────────────────────────────────────────────

_IMAGE_SIZES = {"landscape": "1792x1024", "square": "1024x1024", "portrait": "1024x1792"}


def _is_edit_model(m: dict[str, Any]) -> bool:
    return "edit" in m["id"] or "image" in (m.get("input_modalities") or [])


class QubaxImageGenProvider(ImageGenProvider):
    """Text-to-image and image editing through Qubax (/v1/images/generations, /v1/images/edits)."""

    @property
    def name(self) -> str:
        return "qubax"

    @property
    def display_name(self) -> str:
        return "Qubax AI"

    def is_available(self) -> bool:
        return bool(_api_key())

    def get_setup_schema(self) -> dict[str, Any]:
        return _setup_schema("Qubax AI", "50+ image models (FLUX, Seedream, Nano Banana, GPT Image…), live catalog")

    def list_models(self) -> list[dict[str, Any]]:
        return [_row(m) for m in _models("image")]

    def default_model(self) -> Optional[str]:
        rows = _models("image")
        return _first(rows, lambda m: not _is_edit_model(m)) or (rows[0]["id"] if rows else None)

    def capabilities(self) -> dict[str, Any]:
        return {"modalities": ["text", "image"], "max_reference_images": 7}

    def generate(self, prompt: str, aspect_ratio: str = "landscape", *, image_url: Optional[str] = None,
                 reference_image_urls: Optional[list[str]] = None, **kwargs: Any) -> dict[str, Any]:
        import httpx

        prompt = (prompt or "").strip()
        aspect = resolve_aspect_ratio(aspect_ratio)
        if not prompt:
            return image_error(error="prompt is required", error_type="invalid_request", provider="qubax",
                               aspect_ratio=aspect)
        if not _api_key():
            return image_error(error=f"{KEY_ENV} is not set. Get a key at {SIGNUP_URL}.",
                               error_type="auth_required", provider="qubax", prompt=prompt, aspect_ratio=aspect)
        sources = [s for s in [image_url, *(reference_image_urls or [])] if isinstance(s, str) and s.strip()][:8]
        rows = _models("image")
        by_id = {m["id"]: m for m in rows}
        model = kwargs.get("model") or _configured_model("image_gen") or self.default_model()
        if sources and model in by_id and not _is_edit_model(by_id[model]):
            model = _first(rows, _is_edit_model) or model  # editing needs an edit model
        if not model:
            return image_error(error="No Qubax image model available (catalog unreachable).",
                               error_type="no_model_available", provider="qubax", prompt=prompt,
                               aspect_ratio=aspect)
        body: dict[str, Any] = {"model": model, "prompt": prompt, "n": 1, "response_format": "b64_json"}
        if sources:
            try:
                body["images"] = [_to_data_url(s) for s in sources]
            except ValueError as exc:
                return image_error(error=str(exc), error_type="invalid_request", provider="qubax", model=model,
                                   prompt=prompt, aspect_ratio=aspect)
            path = "/images/edits"
        else:
            body["size"] = _IMAGE_SIZES.get(aspect, "1024x1024")
            path = "/images/generations"
        try:
            resp = httpx.post(_base_url() + path, headers=_headers(), json=body, timeout=300.0)
        except Exception as exc:
            return image_error(error=f"Qubax image request failed: {type(exc).__name__}", error_type="api_error",
                               provider="qubax", model=model, prompt=prompt, aspect_ratio=aspect)
        if resp.status_code >= 400:
            return image_error(error=f"Qubax image generation failed ({_error_text(resp)})",
                               error_type="api_error", provider="qubax", model=model, prompt=prompt,
                               aspect_ratio=aspect)
        data = (resp.json() or {}).get("data") or []
        first = data[0] if data else {}
        prefix = f"qubax_{model.replace('/', '_').replace(':', '_')}"
        try:
            if first.get("b64_json"):
                ref = str(save_b64_image(first["b64_json"], prefix=prefix))
            elif first.get("url"):
                try:
                    ref = str(save_url_image(first["url"], prefix=prefix))
                except Exception:
                    ref = first["url"]
            else:
                return image_error(error="Qubax returned no image.", error_type="empty_response",
                                   provider="qubax", model=model, prompt=prompt, aspect_ratio=aspect)
        except Exception as exc:
            return image_error(error=f"Could not save image: {exc}", error_type="io_error", provider="qubax",
                               model=model, prompt=prompt, aspect_ratio=aspect)
        return image_success(image=ref, model=model, prompt=prompt, aspect_ratio=aspect, provider="qubax",
                             modality="image" if sources else "text")


# ── video ──────────────────────────────────────────────────────────────────

def _is_i2v(m: dict[str, Any]) -> bool:
    caps = m.get("capabilities") or {}
    mid = m["id"]
    return bool(caps.get("imageToVideo")) or "image-to-video" in mid or "i2v" in mid


def _is_t2v(m: dict[str, Any]) -> bool:
    mid = m["id"]
    return "text-to-video" in mid or "t2v" in mid or not _is_i2v(m)


def _pick_duration(m: dict[str, Any], want: Optional[int]) -> Optional[int]:
    allowed = []
    for d in (m.get("capabilities") or {}).get("durations") or []:
        try:
            allowed.append(int(str(d).rstrip("s")))
        except ValueError:
            pass
    if not allowed:
        return want
    allowed.sort()
    if want is None:
        return allowed[0]  # shortest = cheapest
    under = [d for d in allowed if d <= want]
    return under[-1] if under else allowed[0]


def _is_qubax_url(url: str) -> bool:
    """True only for https URLs on the configured Qubax API host (the key is never sent elsewhere)."""
    from urllib.parse import urlparse

    u, base = urlparse(url), urlparse(_base_url())
    return u.scheme == "https" and bool(u.hostname) and u.hostname == base.hostname


def _download_video(url: str, *, prefix: str) -> str:
    """Save the finished video locally; return its path (or the URL if saving fails).

    Qubax's own media URLs are private and need the API key, so they are fetched here with
    the key. Any other host goes through Hermes's SSRF-guarded downloader, without the key."""
    import httpx

    from agent.video_gen_provider import save_bytes_video

    if not _is_qubax_url(url):
        try:
            return str(save_url_video(url, prefix=prefix))
        except Exception as exc:
            logger.debug("qubax-media: caching video failed (%s); returning URL", exc)
            return url
    try:
        with httpx.Client(timeout=180.0, follow_redirects=False) as client:
            r = client.get(url, headers=_headers(json_body=False))
        ctype = (r.headers.get("content-type") or "").split(";")[0].strip()
        if r.status_code != 200 or not ctype.startswith("video/") or not r.content:
            raise RuntimeError(f"HTTP {r.status_code} {ctype}")
        if len(r.content) > 200 * 1024 * 1024:
            raise RuntimeError("video larger than 200 MB")
        ext = {"video/webm": "webm", "video/quicktime": "mov"}.get(ctype, "mp4")
        return str(save_bytes_video(r.content, prefix=prefix, extension=ext))
    except Exception as exc:
        logger.warning("qubax-media: could not download the video (%s). Open it with your key: %s", exc, url)
        return url


class QubaxVideoGenProvider(VideoGenProvider):
    """Text-to-video and image-to-video through Qubax (async job + poll)."""

    @property
    def name(self) -> str:
        return "qubax"

    @property
    def display_name(self) -> str:
        return "Qubax AI"

    def is_available(self) -> bool:
        return bool(_api_key())

    def get_setup_schema(self) -> dict[str, Any]:
        return _setup_schema("Qubax AI", "100+ video models (Seedance, Kling, Wan, Veo…), text & image to video")

    def list_models(self) -> list[dict[str, Any]]:
        return [_row(m) for m in _models("video")]

    def default_model(self) -> Optional[str]:
        rows = _models("video")
        return _first(rows, lambda m: m.get("featured") and _is_t2v(m)) or _first(rows, _is_t2v)

    def capabilities(self) -> dict[str, Any]:
        return {"modalities": ["text", "image"], "aspect_ratios": ["16:9", "9:16", "1:1", "4:3", "3:4", "21:9"],
                "resolutions": ["480p", "720p", "1080p"], "max_duration": 15, "min_duration": 1,
                "supports_audio": True, "supports_negative_prompt": True, "supports_seed": True,
                "supports_upscale": False, "max_reference_images": 0}

    def generate(self, prompt: str, *, model: Optional[str] = None, image_url: Optional[str] = None,
                 reference_image_urls: Optional[list[str]] = None, duration: Optional[int] = None,
                 aspect_ratio: str = "16:9", resolution: str = "720p", negative_prompt: Optional[str] = None,
                 audio: Optional[bool] = None, seed: Optional[int] = None, **kwargs: Any) -> dict[str, Any]:
        import httpx

        prompt = (prompt or "").strip()
        if not _api_key():
            return video_error(error=f"{KEY_ENV} is not set. Get a key at {SIGNUP_URL}.",
                               error_type="missing_credentials", provider="qubax", prompt=prompt)
        rows = _models("video")
        by_id = {m["id"]: m for m in rows}
        model = model or _configured_model("video_gen") or self.default_model()
        if image_url and model in by_id and not _is_i2v(by_id[model]):
            model = _first(rows, lambda m: m.get("featured") and _is_i2v(m)) or _first(rows, _is_i2v) or model
        if not model:
            return video_error(error="No Qubax video model available (catalog unreachable).",
                               error_type="no_model_available", provider="qubax", prompt=prompt)
        meta = by_id.get(model, {})
        caps = meta.get("capabilities") or {}
        body: dict[str, Any] = {"model": model, "prompt": prompt}
        if image_url:
            try:
                body["image_url"] = _to_data_url(image_url)
            except ValueError as exc:
                return video_error(error=str(exc), error_type="invalid_request", provider="qubax", model=model,
                                   prompt=prompt)
        if (dur := _pick_duration(meta, duration)) is not None:
            body["duration"] = dur
        ratios = caps.get("aspectRatios") or []
        if aspect_ratio and (not ratios or aspect_ratio in ratios):
            body["aspect_ratio"] = aspect_ratio
        res = caps.get("resolutions") or []
        if resolution and (not res or resolution in res):
            body["resolution"] = resolution
        if negative_prompt:
            body["negative_prompt"] = negative_prompt
        if seed is not None:
            body["seed"] = seed
        if audio is not None:
            body["generate_audio"] = bool(audio)

        def fail(msg: str, kind: str = "api_error") -> dict[str, Any]:
            return video_error(error=msg, error_type=kind, provider="qubax", model=model, prompt=prompt,
                               aspect_ratio=body.get("aspect_ratio", ""))

        try:
            resp = httpx.post(_base_url() + "/video/generations", headers=_headers(), json=body, timeout=120.0)
        except Exception as exc:
            return fail(f"Qubax video request failed: {type(exc).__name__}")
        if resp.status_code >= 400:
            return fail(f"Qubax video submit failed ({_error_text(resp)})")
        job = resp.json() or {}
        gen_id = job.get("generation_id") or job.get("id")
        if not gen_id:
            return fail("Qubax did not return a job id.", "empty_response")
        status_url = _base_url() + f"/video/generations/{gen_id}"
        deadline = time.monotonic() + _VIDEO_DEADLINE_S
        state: dict[str, Any] = job
        while time.monotonic() < deadline:
            time.sleep(_VIDEO_POLL_S)
            try:
                r = httpx.get(status_url, headers=_headers(json_body=False), timeout=60.0)
            except Exception:
                continue
            if r.status_code >= 400:
                return fail(f"Qubax video status failed ({_error_text(r)})")
            state = r.json() or {}
            status = str(state.get("status") or "").lower()
            if status in ("completed", "succeeded"):
                break
            if status in ("failed", "expired", "cancelled", "canceled"):
                return fail(f"Qubax video job {status}. You are not charged for failed jobs.", "generation_failed")
        else:
            return fail(f"Video still rendering after {_VIDEO_DEADLINE_S // 60} min. "
                        f"Check later: GET {status_url}", "timeout")
        url = state.get("video_url") or ((state.get("result") or {}).get("artifacts") or [{}])[0].get("url")
        if not url:
            return fail("Qubax finished the job but returned no video URL.", "empty_response")
        ref = _download_video(url, prefix=f"qubax_{model}")
        return video_success(video=ref, model=model, prompt=prompt, modality="image" if image_url else "text",
                             aspect_ratio=body.get("aspect_ratio", ""), duration=body.get("duration") or 0,
                             provider="qubax", extra={"generation_id": gen_id})


# ── text-to-speech ─────────────────────────────────────────────────────────

_TTS_FORMATS = {"mp3", "wav", "opus", "flac", "aac", "pcm"}


class QubaxTTSProvider(TTSProvider):
    """Text-to-speech through Qubax (/v1/audio/speech)."""

    @property
    def name(self) -> str:
        return "qubax"

    @property
    def display_name(self) -> str:
        return "Qubax AI"

    def is_available(self) -> bool:
        return bool(_api_key())

    def get_setup_schema(self) -> dict[str, Any]:
        return _setup_schema("Qubax AI", "ElevenLabs, MiniMax, Kokoro, OpenAI TTS and more, one key")

    def list_models(self) -> list[dict[str, Any]]:
        return [_row(m) for m in _models("tts")]

    def default_model(self) -> Optional[str]:
        rows = _models("tts")
        return _first(rows) or (rows[0]["id"] if rows else None)

    def synthesize(self, text: str, output_path: str, *, voice: Optional[str] = None, model: Optional[str] = None,
                   speed: Optional[float] = None, format: str = "mp3", **extra: Any) -> str:
        import httpx

        if not _api_key():
            raise RuntimeError(f"{KEY_ENV} is not set. Get a key at {SIGNUP_URL}.")
        model = model or self.default_model()
        if not model:
            raise RuntimeError("No Qubax TTS model available (catalog unreachable).")
        fmt = (format or "mp3").lower()
        fmt = "opus" if fmt == "ogg" else fmt
        fmt = fmt if fmt in _TTS_FORMATS else "mp3"
        body: dict[str, Any] = {"model": model, "input": text, "response_format": fmt}
        if voice:
            body["voice"] = voice  # only when set: many models reject unknown default voices
        if speed:
            body["speed"] = speed
        resp = httpx.post(_base_url() + "/audio/speech", headers=_headers(), json=body, timeout=180.0)
        if resp.status_code >= 400:
            raise RuntimeError(f"Qubax TTS failed ({_error_text(resp)})")
        if not resp.content:
            raise RuntimeError("Qubax TTS returned no audio.")
        out = Path(output_path)
        want_ext = ".ogg" if fmt == "opus" else f".{fmt}"
        if out.suffix.lower() not in (want_ext, f".{fmt}"):
            out = out.with_suffix(want_ext)
        out.write_bytes(resp.content)
        return str(out)


# ── speech-to-text ─────────────────────────────────────────────────────────

class QubaxTranscriptionProvider(TranscriptionProvider):
    """Speech-to-text through Qubax (/v1/audio/transcriptions)."""

    @property
    def name(self) -> str:
        return "qubax"

    @property
    def display_name(self) -> str:
        return "Qubax AI"

    def is_available(self) -> bool:
        return bool(_api_key())

    def get_setup_schema(self) -> dict[str, Any]:
        return _setup_schema("Qubax AI", "Whisper, ElevenLabs Scribe, xAI and more, one key")

    def list_models(self) -> list[dict[str, Any]]:
        return [_row(m) for m in _models("stt")]

    def default_model(self) -> Optional[str]:
        rows = _models("stt")
        return _first(rows) or (rows[0]["id"] if rows else None)

    def transcribe(self, file_path: str, *, model: Optional[str] = None, language: Optional[str] = None,
                   **extra: Any) -> dict[str, Any]:
        import httpx

        if not _api_key():
            return {"success": False, "transcript": "", "error": f"{KEY_ENV} is not set.", "provider": "qubax"}
        model = model or self.default_model()
        if not model:
            return {"success": False, "transcript": "", "error": "No Qubax speech-to-text model available.",
                    "provider": "qubax"}
        data: dict[str, Any] = {"model": model, "response_format": "json"}
        if language:
            data["language"] = language
        if isinstance(extra.get("prompt"), str) and extra["prompt"].strip():
            data["prompt"] = extra["prompt"].strip()
        try:
            p = Path(file_path)
            with p.open("rb") as fh:
                resp = httpx.post(_base_url() + "/audio/transcriptions", headers=_headers(json_body=False),
                                  data=data, timeout=300.0,
                                  files={"file": (p.name, fh, mimetypes.guess_type(p.name)[0] or "audio/mpeg")})
        except Exception as exc:
            return {"success": False, "transcript": "", "error": f"Qubax STT request failed: {type(exc).__name__}",
                    "provider": "qubax"}
        if resp.status_code >= 400:
            return {"success": False, "transcript": "", "error": f"Qubax STT failed ({_error_text(resp)})",
                    "provider": "qubax"}
        try:
            payload = resp.json()
            text = payload.get("text", "") if isinstance(payload, dict) else str(payload)
        except Exception:
            text = resp.text
        text = (text or "").strip()
        if not text:
            return {"success": False, "transcript": "", "error": "Qubax returned an empty transcript.",
                    "provider": "qubax"}
        return {"success": True, "transcript": text, "provider": "qubax", "model": model}


def register(ctx) -> None:
    """Plugin entry point: register Qubax as image, video, TTS and STT backend."""
    ctx.register_image_gen_provider(QubaxImageGenProvider())
    ctx.register_video_gen_provider(QubaxVideoGenProvider())
    ctx.register_tts_provider(QubaxTTSProvider())
    ctx.register_transcription_provider(QubaxTranscriptionProvider())
