"""Judge checks before a local run: a bad key, an unknown model or a dead endpoint fail in seconds, with a hint."""
from __future__ import annotations

import httpx
import pytest

from supposing.cli import main, preflight
from supposing.cli.output import CliError
from supposing.cloud import JudgeError, parse_judge


@pytest.fixture
def web(monkeypatch):
    """httpx.get answered by a table of url -> status (or an exception, or a JSON body)."""
    answers: dict[str, object] = {}

    def get(url, headers=None, timeout=None):
        for prefix, ans in answers.items():
            if url.startswith(prefix):
                if isinstance(ans, Exception):
                    raise ans
                if isinstance(ans, dict):
                    return httpx.Response(200, json=ans)
                return httpx.Response(int(ans))  # type: ignore[arg-type]
        raise httpx.ConnectError("no route")
    monkeypatch.setattr(preflight.httpx, "get", get)
    return answers


def error(fn) -> CliError:
    with pytest.raises(CliError) as exc:
        fn()
    return exc.value


def test_openrouter_bad_key_and_unknown_model(web, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-fake")
    web["https://openrouter.ai/api/v1/key"] = 401
    e = error(lambda: preflight.check("openrouter", "anthropic/claude-haiku-5.5"))
    assert "refused the key" in e.message and "openrouter.ai/settings/keys" in (e.hint or "")
    web["https://openrouter.ai/api/v1/key"] = 200
    web["https://openrouter.ai/api/v1/models/acme/nope"] = 404
    e = error(lambda: preflight.check("openrouter", "acme/nope"))
    assert "no model acme/nope" in e.message and "--judge quick" in (e.hint or "")
    web["https://openrouter.ai/api/v1/models/"] = 200
    preflight.check("openrouter", "anthropic/claude-haiku-5.5")  # fine: no error


def test_a_dead_or_wrong_local_endpoint(web):
    url = "http://localhost:11434/v1"
    e = error(lambda: preflight.check("openai_compat", "llama3.2", url))
    assert "Can't reach" in e.message and "ollama serve" in (e.hint or "") and url in (e.hint or "")
    web[url + "/models"] = {"data": [{"id": "llama3.2:latest"}, {"id": "qwen3:8b"}]}
    preflight.check("openai_compat", "llama3.2", url)  # the :latest tag counts
    e = error(lambda: preflight.check("openai_compat", "llama9", url))
    assert "No model llama9" in e.message and "qwen3:8b" in (e.hint or "") and "ollama pull llama9" in (e.hint or "")
    web[url + "/models"] = 404  # a server with no model list: let the run decide
    preflight.check("openai_compat", "anything", url)


def test_offline_checks_never_block(web, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-x")
    preflight.check("openrouter", "anthropic/claude-haiku-5.5")  # every request fails to connect: no error


def test_judge_messages_fit_the_judge():
    with pytest.raises(JudgeError, match="claude-haiku-5-5"):
        parse_judge("anthropic", local=True)
    with pytest.raises(JudgeError, match="llama3.2"):
        parse_judge("openai_compat", local=True)
    with pytest.raises(JudgeError, match="Did you mean mock"):
        parse_judge("mok", local=True)


def test_rank_stops_before_asking_when_the_endpoint_is_down(web, tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("SUPPOSING_CONFIG", str(tmp_path / "c.json"))
    for i in range(3):
        (tmp_path / ("d%d.md" % i)).write_text("doc %d" % i)
    code = main(["rank", str(tmp_path), "--rubric", "x", "--judge", "openai_compat", "--model", "llama3.2",
                 "--base-url", "http://localhost:11434/v1", "--json"])
    assert code == 2 and "ollama serve" in capsys.readouterr().out


def test_no_key_names_the_free_judges(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    for i in range(3):
        (tmp_path / ("d%d.md" % i)).write_text("doc %d" % i)
    main(["rank", str(tmp_path), "--rubric", "x", "--json"])
    hint = capsys.readouterr().out
    assert "--judge claude_cli" in hint and "openai_compat" in hint and "--judge mock" in hint
