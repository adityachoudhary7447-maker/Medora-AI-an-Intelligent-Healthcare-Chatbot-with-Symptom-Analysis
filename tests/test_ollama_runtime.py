import json
from urllib.error import URLError
from unittest.mock import Mock, patch

import pytest

from src import ollama_runtime


@patch("src.ollama_runtime.urlopen")
def test_list_installed_models_reads_local_tags(mock_urlopen):
    response = Mock()
    response.read.return_value = json.dumps({
        "models": [{"name": "llama3.2:latest"}, {"name": "llama3.2:1b"}]
    }).encode()
    mock_urlopen.return_value.__enter__.return_value = response

    assert ollama_runtime.list_installed_models() == ["llama3.2:latest", "llama3.2:1b"]
    mock_urlopen.assert_called_once()


@patch("src.ollama_runtime.urlopen", side_effect=URLError("connection refused"))
def test_list_installed_models_explains_unavailable_ollama(mock_urlopen):
    with pytest.raises(ollama_runtime.OllamaRuntimeError, match="Start Ollama"):
        ollama_runtime.list_installed_models()


@patch("src.ollama_runtime.list_installed_models", return_value=["llama3.2:latest"])
def test_require_model_available_accepts_implicit_latest(mock_list_models):
    ollama_runtime.require_model_available("llama3.2")


@patch("src.ollama_runtime.list_installed_models", return_value=["llama3.2:latest"])
def test_require_model_available_suggests_pull_for_missing_model(mock_list_models):
    with pytest.raises(ollama_runtime.OllamaRuntimeError, match="ollama pull qwen2.5vl:3b"):
        ollama_runtime.require_model_available("qwen2.5vl:3b")


def test_remote_ollama_endpoints_are_rejected():
    with pytest.raises(ollama_runtime.OllamaRuntimeError, match="remote endpoints"):
        ollama_runtime._validate_local_url("https://example.com")


@patch("src.ollama_runtime.require_model_available")
@patch("src.ollama_runtime.get_model_capabilities", return_value=["completion", "tools"])
@patch("src.ollama_runtime.available_memory_gb", return_value=1.2)
def test_vision_model_is_blocked_without_vision_capability(
    mock_memory,
    mock_capabilities,
    mock_require_model,
):
    with pytest.raises(ollama_runtime.OllamaRuntimeError, match="does not report vision"):
        ollama_runtime.require_vision_resources("llama3.2")


@patch("src.ollama_runtime.require_model_available")
@patch("src.ollama_runtime.get_model_capabilities", return_value=["completion", "vision"])
@patch("src.ollama_runtime.available_memory_gb", return_value=1.2)
def test_vision_model_is_blocked_without_memory_headroom(
    mock_memory,
    mock_capabilities,
    mock_require_model,
):
    with pytest.raises(ollama_runtime.OllamaRuntimeError, match="at least 4 GiB"):
        ollama_runtime.require_vision_resources("qwen2.5vl:3b")


def test_memory_errors_get_actionable_local_guidance():
    message = ollama_runtime.format_ollama_error(
        RuntimeError("model requires more system memory (3.4 GiB)"),
        "llama3.2",
    )

    assert "smaller local model" in message


def test_timeouts_get_local_runtime_guidance():
    message = ollama_runtime.format_ollama_error(TimeoutError("request timed out"))

    assert "local Ollama request timed out" in message
    assert "try again" in message
