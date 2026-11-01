"""Local Ollama configuration and readiness checks."""

import ctypes
import json
import os
import sys
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b").strip()
OLLAMA_VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL", "").strip() or None
OLLAMA_REQUEST_TIMEOUT = float(os.getenv("OLLAMA_REQUEST_TIMEOUT", "120"))
VISION_MIN_AVAILABLE_MEMORY_GB = 4.0
_OLLAMA_HEALTH_TIMEOUT = 5
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "host.docker.internal"}


class OllamaRuntimeError(RuntimeError):
    """Raised when the configured local Ollama runtime is not ready."""


def _validate_local_url(base_url: str) -> str:
    parsed = urlsplit(base_url)
    if parsed.scheme != "http" or parsed.hostname not in _LOCAL_HOSTS:
        raise OllamaRuntimeError(
            "Ollama must use a local HTTP endpoint such as "
            "http://localhost:11434; remote endpoints are not supported."
        )
    if parsed.username or parsed.password:
        raise OllamaRuntimeError("Ollama URLs must not contain credentials.")
    return base_url.rstrip("/")


def list_installed_models(base_url: str = OLLAMA_BASE_URL) -> list[str]:
    """Return local model names, or raise a diagnostic for an unavailable Ollama."""
    local_url = _validate_local_url(base_url)
    request = Request(f"{local_url}/api/tags")

    try:
        with urlopen(request, timeout=_OLLAMA_HEALTH_TIMEOUT) as response:
            data = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        raise OllamaRuntimeError(
            f"Ollama returned HTTP {error.code} while listing local models."
        ) from error
    except (URLError, TimeoutError, OSError) as error:
        raise OllamaRuntimeError(
            f"Cannot connect to local Ollama at {local_url}. Start Ollama and retry."
        ) from error
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise OllamaRuntimeError(
            "Ollama returned an invalid response from its local model-list endpoint."
        ) from error

    models = data.get("models")
    if not isinstance(models, list):
        raise OllamaRuntimeError("Ollama's local model list has an unexpected format.")

    return [
        model["name"]
        for model in models
        if isinstance(model, dict) and isinstance(model.get("name"), str)
    ]


def require_model_available(
    model_name: str,
    base_url: str = OLLAMA_BASE_URL,
) -> None:
    """Raise an actionable error unless the requested model is already installed."""
    if not model_name:
        raise OllamaRuntimeError("No Ollama model name is configured.")

    installed = list_installed_models(base_url)
    normalized_name = (
        model_name if ":" in model_name.rsplit("/", 1)[-1]
        else f"{model_name}:latest"
    )
    normalized_installed = {
        name if ":" in name.rsplit("/", 1)[-1] else f"{name}:latest"
        for name in installed
    }

    if normalized_name not in normalized_installed:
        installed_text = ", ".join(installed) if installed else "none"
        raise OllamaRuntimeError(
            f"Local Ollama model '{model_name}' is not installed. "
            f"Run `ollama pull {model_name}`. Installed models: {installed_text}."
        )


def get_model_capabilities(
    model_name: str,
    base_url: str = OLLAMA_BASE_URL,
) -> list[str]:
    """Return Ollama's reported capabilities for an installed local model."""
    local_url = _validate_local_url(base_url)
    request = Request(
        f"{local_url}/api/show",
        data=json.dumps({"name": model_name}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=_OLLAMA_HEALTH_TIMEOUT) as response:
            data = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        if error.code == 404:
            raise OllamaRuntimeError(
                f"Local Ollama model '{model_name}' is not installed. "
                f"Run `ollama pull {model_name}`."
            ) from error
        raise OllamaRuntimeError(
            f"Ollama returned HTTP {error.code} while checking model capabilities."
        ) from error
    except (URLError, TimeoutError, OSError) as error:
        raise OllamaRuntimeError(
            f"Cannot connect to local Ollama at {local_url}. Start Ollama and retry."
        ) from error
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise OllamaRuntimeError(
            "Ollama returned an invalid response while checking model capabilities."
        ) from error

    capabilities = data.get("capabilities")
    if not isinstance(capabilities, list) or not all(
        isinstance(capability, str) for capability in capabilities
    ):
        raise OllamaRuntimeError(
            f"Ollama did not report valid capabilities for '{model_name}'."
        )

    return capabilities


def available_memory_gb() -> Optional[float]:
    """Return currently available physical memory when the platform exposes it."""
    if sys.platform == "win32":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return status.ullAvailPhys / (1024 ** 3)
        return None

    if hasattr(os, "sysconf"):
        try:
            available_pages = os.sysconf("SC_AVPHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            return available_pages * page_size / (1024 ** 3)
        except (ValueError, OSError):
            return None

    return None


def require_vision_resources(
    model_name: str,
    base_url: str = OLLAMA_BASE_URL,
) -> None:
    """Check that an optional local vision model is installed and has RAM headroom."""
    require_model_available(model_name, base_url)
    capabilities = get_model_capabilities(model_name, base_url)
    if "vision" not in capabilities:
        raise OllamaRuntimeError(
            f"Local Ollama model '{model_name}' does not report vision capability. "
            "Choose an installed image-capable model before enabling image analysis."
        )

    available = available_memory_gb()
    if available is not None and available < VISION_MIN_AVAILABLE_MEMORY_GB:
        raise OllamaRuntimeError(
            f"Local image analysis needs at least "
            f"{VISION_MIN_AVAILABLE_MEMORY_GB:.0f} GiB of available RAM; "
            f"this computer currently has {available:.1f} GiB available. "
            "Close other applications or use a computer with more memory."
        )


def format_ollama_error(error: Exception, model_name: str = OLLAMA_MODEL) -> str:
    """Convert common local inference failures into useful user-facing guidance."""
    detail = str(error)
    lowered = detail.lower()

    if "timed out" in lowered or "timeout" in lowered:
        return (
            "The local Ollama request timed out. The first response can take longer "
            "while the model loads; check that Ollama is running and try again."
        )
    if any(
        message in lowered
        for message in (
            "connection refused",
            "failed to establish a new connection",
            "connecterror",
            "all connection attempts failed",
        )
    ):
        return (
            f"Cannot connect to local Ollama at {OLLAMA_BASE_URL}. "
            "Start Ollama and retry."
        )
    if "requires more system memory" in lowered or "runner process has terminated" in lowered:
        return (
            f"Ollama could not load '{model_name}' with the memory currently available. "
            "Close other applications and retry, or choose a smaller local model "
            "after checking its hardware requirements."
        )
    if "model" in lowered and ("not found" in lowered or "not installed" in lowered):
        return f"Local model '{model_name}' is missing. Install it with `ollama pull {model_name}`."

    return f"Local Ollama request failed: {detail}"
