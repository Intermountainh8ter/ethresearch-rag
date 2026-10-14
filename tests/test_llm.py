import sys
import types

import pytest

from ethrag.llm import DEFAULT_MODEL, AnthropicLLM, ScriptedLLM, api_key_available


def test_scripted_llm_records_calls_and_runs_out():
    llm = ScriptedLLM(["one", "two"])
    assert llm.complete("s", "u1") == "one" and llm.complete("s", "u2") == "two"
    assert llm.calls == [("s", "u1"), ("s", "u2")]
    with pytest.raises(RuntimeError, match="ran out"):
        llm.complete("s", "u3")


def test_api_key_available(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert api_key_available() is False
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    assert api_key_available() is True


def _fake_anthropic(monkeypatch, seen):
    class Block:
        def __init__(self, type, text=""):
            self.type, self.text = type, text

    class Messages:
        def create(self, **kw):
            seen.update(kw)
            return types.SimpleNamespace(content=[Block("text", "hel"), Block("tool_use"), Block("text", "lo")])

    class Client:
        messages = Messages()

    mod = types.ModuleType("anthropic")
    mod.Anthropic = Client
    monkeypatch.setitem(sys.modules, "anthropic", mod)


def test_anthropic_llm_with_a_stubbed_sdk(monkeypatch):
    seen = {}
    _fake_anthropic(monkeypatch, seen)
    monkeypatch.delenv("ETHRAG_MODEL", raising=False)
    llm = AnthropicLLM(max_tokens=123)
    assert llm.model == DEFAULT_MODEL
    assert llm.complete("SYS", "USER") == "hello"  # only text blocks are joined
    assert seen["system"] == "SYS" and seen["max_tokens"] == 123
    assert seen["messages"] == [{"role": "user", "content": "USER"}]


def test_model_comes_from_arg_then_env_then_default(monkeypatch):
    _fake_anthropic(monkeypatch, {})
    monkeypatch.setenv("ETHRAG_MODEL", "from-env")
    assert AnthropicLLM().model == "from-env"
    assert AnthropicLLM(model="explicit").model == "explicit"
