"""Qubax AI provider profile for Hermes Agent.

Qubax (https://qubax.ai) is an OpenAI-compatible API with 400+ models,
often far below list price. This plugin adds "Qubax" to the Hermes provider
list; the picker shows every Qubax text model, featured ones first.

Everything model-related is read from the live public catalog
(``/v1/models``), so prices, new models and retirements track Qubax without
a plugin update. Nothing here sends data anywhere except api.qubax.ai.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import urllib.request
from typing import Any

from providers import register_provider
from providers.base import ProviderProfile

logger = logging.getLogger(__name__)

BASE_URL = "https://api.qubax.ai/v1"
MODELS_URL = BASE_URL + "/models"
BALANCE_URL = BASE_URL + "/billing/balance"

# Used only when the live catalog cannot be reached. All are featured,
# text-out, tool-capable Qubax models (checked Nov 2026); cheapest first.
FALLBACK_MODELS = (
    "glm-5.3-flash",
    "qwen-3-8-27b",
    "deepseek-v4.1-flash-1017",
    "claude-haiku-5.5",
    "gpt-6.1-sol",
    "claude-opus-5.5",
)
DEFAULT_CHEAP_MODEL = "glm-5.3-flash"

_CATALOG_TTL_S = 15 * 60
_EFFORT_LADDER = ("minimal", "low", "medium", "high", "xhigh", "max")

_lock = threading.Lock()
_catalog: dict[str, dict[str, Any]] | None = None
_catalog_at = 0.0


def _user_agent() -> str:
    try:
        from hermes_cli.version_info import hermes_user_agent

        return hermes_user_agent("hermes-qubax")
    except Exception:
        return "hermes-qubax/1.0"


def _load_catalog(timeout: float = 8.0, *, force: bool = False) -> dict[str, dict[str, Any]]:
    """Live public catalog keyed by model id, cached for 15 min. Never raises; {} on failure."""
    global _catalog, _catalog_at
    with _lock:
        if not force and _catalog is not None and time.monotonic() - _catalog_at < _CATALOG_TTL_S:
            return _catalog
    try:
        req = urllib.request.Request(MODELS_URL, headers={"Accept": "application/json", "User-Agent": _user_agent()})
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # public, no key needed
            items = json.loads(resp.read().decode()).get("data") or []
        catalog = {m["id"]: m for m in items if isinstance(m, dict) and m.get("id")}
    except Exception as exc:
        logger.debug("qubax: catalog fetch failed: %s", exc)
        with _lock:
            return _catalog or {}
    with _lock:
        _catalog, _catalog_at = catalog, time.monotonic()
    return catalog


def _is_chat(m: dict[str, Any]) -> bool:
    """Text out, text in, tool calling (Hermes is an agent: no tools = no use)."""
    outs = m.get("output_modalities") or []
    ins = m.get("input_modalities") or []
    params = m.get("supported_parameters") or []
    return outs == ["text"] and "text" in ins and "tools" in params and m.get("is_ready", True) is not False


def _price(m: dict[str, Any]) -> float:
    p = m.get("pricing_summary") or {}
    try:
        return float(p.get("prompt_per_1m") or 0) + float(p.get("completion_per_1m") or 0)
    except (TypeError, ValueError):
        return float("inf")


def featured_chat_models(catalog: dict[str, dict[str, Any]] | None = None) -> list[str]:
    """Featured text models with tools, cheapest first (ties: most used)."""
    catalog = _load_catalog() if catalog is None else catalog
    rows = [m for m in catalog.values() if m.get("featured") and _is_chat(m)]
    rows.sort(key=lambda m: (_price(m), -(m.get("requests_24h") or 0)))
    return [m["id"] for m in rows]


def all_chat_models(catalog: dict[str, dict[str, Any]] | None = None) -> list[str]:
    """Every text model with tools: featured first (cheapest first), then the rest (cheapest first)."""
    catalog = _load_catalog() if catalog is None else catalog
    featured = featured_chat_models(catalog)
    seen = set(featured)
    rest = [m for m in catalog.values() if m["id"] not in seen and _is_chat(m)]
    rest.sort(key=lambda m: (_price(m), -(m.get("requests_24h") or 0)))
    return featured + [m["id"] for m in rest]


def _cheapest(catalog: dict[str, dict[str, Any]], *, vision: bool = False) -> str:
    for mid in featured_chat_models(catalog):
        m = catalog[mid]
        if _price(m) <= 0:
            continue  # skip $0 rows (free promos come and go)
        if vision and "image" not in (m.get("input_modalities") or []):
            continue
        return mid
    return ""


class QubaxProfile(ProviderProfile):
    """Qubax: OpenAI-compatible, live catalog, per-model reasoning levels."""

    # Picker: every text model with tools. Featured first, then the rest; each group cheapest first.
    def fetch_models(self, *, api_key: str | None = None, base_url: str | None = None,
                     timeout: float = 8.0) -> list[str] | None:
        custom = (base_url or "").strip().rstrip("/")
        if custom and custom != BASE_URL:
            return super().fetch_models(api_key=api_key, base_url=base_url, timeout=timeout)
        ids = all_chat_models(_load_catalog(timeout=timeout))
        return ids or None

    def invalidate_catalog_cache(self) -> None:
        global _catalog
        with _lock:
            _catalog = None

    # Cheap model for compression / titles / summaries: cheapest featured text model.
    def resolve_aux_model(self, *, vision: bool = False) -> str:
        try:
            return _cheapest(_load_catalog(timeout=4.0), vision=vision)
        except Exception:
            return ""

    def default_vision_model(self) -> str | None:
        return _cheapest(_load_catalog(timeout=4.0), vision=True) or None

    # Context window straight from Qubax's catalog (models.dev doesn't know our ids).
    def live_context_length(self, model: str, *, api_key: str = "", base_url: str = "") -> int | None:
        m = _load_catalog(timeout=4.0).get(model) or {}
        ctx = m.get("context_length")
        return int(ctx) if isinstance(ctx, int) and ctx > 0 else None

    def get_model_context_length(self, model: str) -> int | None:
        return self.live_context_length(model)

    def supported_reasoning_efforts(self, model: str | None) -> tuple[str, ...] | None:
        m = _load_catalog(timeout=4.0).get(model or "") or {}
        levels = (m.get("capabilities") or {}).get("reasoningEffortLevels")
        if not isinstance(levels, list):
            return None
        return tuple(lvl for lvl in levels if lvl in _EFFORT_LADDER)

    def build_api_kwargs_extras(self, *, reasoning_config: dict | None = None, **context: Any):
        """Send ``reasoning_effort`` only when the model publishes levels; clamp to the nearest
        weaker supported level so a request never costs more than the user asked for."""
        if not isinstance(reasoning_config, dict) or reasoning_config.get("enabled") is False:
            return {}, {}
        effort = str(reasoning_config.get("effort") or "").strip().lower()
        if effort not in _EFFORT_LADDER:
            return {}, {}
        supported = self.supported_reasoning_efforts(context.get("model"))
        if not supported:
            return {}, {}  # fixed or unknown: the model's own default applies
        if effort not in supported:
            idx = _EFFORT_LADDER.index(effort)
            weaker = [lvl for lvl in supported if _EFFORT_LADDER.index(lvl) < idx]
            effort = max(weaker, key=_EFFORT_LADDER.index) if weaker else min(supported, key=_EFFORT_LADDER.index)
        return {}, {"reasoning_effort": effort}

    # /usage shows the prepaid balance.
    def fetch_account_usage(self, *, base_url: str | None = None, api_key: str | None = None):
        from datetime import datetime, timezone

        from agent.account_usage import AccountUsageSnapshot
        from hermes_cli.runtime_provider import resolve_runtime_provider

        runtime = resolve_runtime_provider(requested=self.name, explicit_base_url=base_url, explicit_api_key=api_key)
        token = str(runtime.get("api_key", "") or "").strip()
        if not token:
            return None
        req = urllib.request.Request(BALANCE_URL, headers={
            "Authorization": f"Bearer {token}", "Accept": "application/json", "User-Agent": _user_agent()})
        with urllib.request.urlopen(req, timeout=6.0) as resp:
            data = json.loads(resp.read().decode())
        balance = float(data.get("balance_usd") or 0)
        return AccountUsageSnapshot(
            provider=self.name, source="qubax_balance_api", fetched_at=datetime.now(timezone.utc),
            title="Qubax balance", plan="Prepaid credits",
            details=(f"Balance: ${balance:,.2f}", "Top up: https://qubax.ai/app/billing"),
            raw=data,
        )


qubax = QubaxProfile(
    name="qubax",
    aliases=("qubax-ai", "qubaxai"),
    display_name="Qubax AI",
    description="Qubax AI: 400+ models through one API, pay with crypto",
    signup_url="https://qubax.ai/app/api-keys",
    env_vars=("QUBAX_API_KEY", "QUBAX_BASE_URL"),
    base_url=BASE_URL,
    models_url=MODELS_URL,
    auth_type="api_key",
    default_aux_model=DEFAULT_CHEAP_MODEL,
    fallback_models=FALLBACK_MODELS,
    hostnames=("api.qubax.ai",),
    supports_vision=True,
)

register_provider(qubax)
