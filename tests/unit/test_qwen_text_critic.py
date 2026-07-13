"""Unit tests for the Qwen text-critic HTTP client (P4-6) via httpx.MockTransport.

Critics are built over an injected MockTransport (critic-owned) and closed by the
``make_critic`` fixture teardown so no httpx client / event loop leaks — important
because ``filterwarnings = error`` turns a stray ResourceWarning into a test failure.
"""

from __future__ import annotations

import json

import httpx
import pytest
import pytest_asyncio

from my_curator.adapters.judge.qwen_text_critic import (
    JudgeCriticError,
    QwenTextCritic,
    SamplingParams,
    base_url_from_env,
)

pytestmark = pytest.mark.unit


def _ok_response(content: str) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})


@pytest_asyncio.fixture
async def make_critic():
    """Factory that builds critics over a MockTransport and closes them on teardown."""
    created: list[QwenTextCritic] = []

    def _make(handler, *, base_url="http://j", **kw) -> QwenTextCritic:
        critic = QwenTextCritic(base_url, transport=httpx.MockTransport(handler), **kw)
        created.append(critic)
        return critic

    yield _make
    for critic in created:
        await critic.aclose()


async def test_critique_returns_content_and_sends_expected_body(make_critic):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        return _ok_response("VERDICT_RISK: KEEP\nCONFIDENCE: high")

    critic = make_critic(handler, base_url="http://judge:8002")
    out = await critic.critique("SYS", "USER")
    assert "VERDICT_RISK: KEEP" in out
    assert captured["url"] == "http://judge:8002/v1/chat/completions"
    body = captured["body"]
    assert body["model"] == "qwen3-8b-awq"
    assert body["messages"][0] == {"role": "system", "content": "SYS"}
    assert body["messages"][1] == {"role": "user", "content": "USER"}
    # card sampling params present
    assert body["temperature"] == 0.6 and body["top_p"] == 0.95
    assert body["top_k"] == 20 and body["presence_penalty"] == 1.5
    assert body["max_tokens"] == 2048


async def test_retry_then_success(make_critic):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(503, text="unavailable")
        return _ok_response("ok")

    critic = make_critic(handler, retries=1)
    assert await critic.critique("s", "u") == "ok"
    assert calls["n"] == 2  # one retry


async def test_raises_after_exhausting_retries(make_critic):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    critic = make_critic(handler, retries=1)
    with pytest.raises(JudgeCriticError):
        await critic.critique("s", "u")


async def test_timeout_raises_judge_critic_error(make_critic):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("timed out")

    critic = make_critic(handler, retries=0)
    with pytest.raises(JudgeCriticError):
        await critic.critique("s", "u")


async def test_malformed_body_raises(make_critic):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": True})

    critic = make_critic(handler, retries=0)
    with pytest.raises(JudgeCriticError):
        await critic.critique("s", "u")


async def test_custom_sampling_params_serialized(make_critic):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return _ok_response("x")

    critic = make_critic(handler, sampling=SamplingParams(temperature=0.3, max_tokens=512))
    await critic.critique("s", "u")
    assert captured["body"]["temperature"] == 0.3
    assert captured["body"]["max_tokens"] == 512


def test_base_url_from_env(monkeypatch):
    monkeypatch.delenv("JUDGE_CRITIC_URL", raising=False)
    assert base_url_from_env() == "http://localhost:8002"
    monkeypatch.setenv("JUDGE_CRITIC_URL", "http://host:9999")
    assert base_url_from_env() == "http://host:9999"


async def test_context_manager_closes():
    def handler(request: httpx.Request) -> httpx.Response:
        return _ok_response("y")

    async with QwenTextCritic("http://j", transport=httpx.MockTransport(handler)) as critic:
        assert await critic.critique("s", "u") == "y"
