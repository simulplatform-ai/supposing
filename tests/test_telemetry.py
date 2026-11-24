"""CLI telemetry: only allow-listed fields leave the machine, every off switch works, and nothing else sends."""
from __future__ import annotations

import io
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
from pathlib import Path

import pytest

from supposing.cli import main, telemetry
from supposing.cli import telemetry_schema as schema

SECRETS = ("PLANTEDTEXT", "RUBRICSECRET", "TOPSECRETFILE", "secretfolder", "sk-or-SECRETKEY", "acme/secret-model")


@pytest.fixture
def sent(tmp_path, monkeypatch):
    """Telemetry as on a person's terminal after the notice was shown; returns the bodies posted."""
    monkeypatch.setenv("SUPPOSING_CONFIG", str(tmp_path / "cfg" / "config.json"))
    for var in (*telemetry.CI_VARS, "SUPPOSING_TELEMETRY", "DO_NOT_TRACK", "SUPPOSING_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(telemetry, "_under_pytest", lambda: False)
    monkeypatch.setattr(telemetry, "_stdout_is_tty", lambda: True)
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    telemetry.save_state({"notice_shown": "2026-11-24"})
    bodies: list[dict] = []
    monkeypatch.setattr(telemetry, "post", lambda body: bodies.append(json.loads(body)))
    return bodies


@pytest.fixture
def docs(tmp_path):
    d = tmp_path / "secretfolder"
    d.mkdir()
    for i, q in enumerate([0.9, 0.5, 0.1]):
        (d / ("TOPSECRETFILE%d.md" % i)).write_text("PLANTEDTEXT %d [[q=%s]]" % (i, q))
    return d


def rank(docs, *flags: str) -> int:
    return main(["rank", str(docs), "--rubric", "RUBRICSECRET: which is best?", "--rounds", "2", *flags])


def test_an_event_holds_only_allow_listed_fields(sent, docs, capsys):
    assert rank(docs, "--judge", "mock") == 0
    (event,) = sent
    assert set(event) == set(schema.FIELDS) and schema.valid(event), schema.problems(event)
    assert event["command"] == "rank" and event["local"] is True and event["judge"] == "mock"
    assert event["items"] == "1-5" and event["rounds"] == 2 and event["both_orders"] is False
    assert event["outcome"] == "ok" and event["version"] and event["sample"] is None
    assert schema.problems({**event, "cwd": "/home/x"}) == ["unknown field cwd"]
    assert schema.problems({**event, "judge": "acme/secret-model"})
    assert schema.problems({k: v for k, v in event.items() if k != "os"}) == ["missing field os"]


def test_planted_secrets_paths_and_text_never_leave(sent, docs, capsys, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-SECRETKEY")
    monkeypatch.chdir(docs)
    rank(docs, "--judge", "acme/secret-model", "--dry-run")  # a custom model: sent as "other"
    rank(docs, "--judge", "mock")
    main(["rank", "--sample", "stories", "--judge", "mock", "--rounds", "1", "--json"])
    capsys.readouterr()
    body = json.dumps(sent)
    private = (*SECRETS, str(docs), os.getcwd(), socket.gethostname(), Path.home().name, sys.executable)
    assert not [s for s in private if s and s in body], body
    assert sent[0]["judge"] == "openrouter" and sent[0]["model"] == "other"
    assert sent[2]["sample"] == "stories" and sent[2]["items"] == "6-20"
    assert all(schema.valid(e) for e in sent)


@pytest.mark.parametrize("env", [{"SUPPOSING_TELEMETRY": "0"}, {"SUPPOSING_TELEMETRY": "off"}, {"DO_NOT_TRACK": "1"},
                                 {"CI": "true"}, {"GITHUB_ACTIONS": "true"}, {"GITLAB_CI": "1"},
                                 {"BUILDKITE": "true"}, {"JENKINS_URL": "http://ci"}])
def test_env_switches_and_ci_turn_it_off(sent, docs, capsys, monkeypatch, env):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    assert rank(docs, "--judge", "mock") == 0 and sent == []


def test_the_off_command_and_status(sent, docs, capsys):
    assert main(["telemetry", "off"]) == 0
    rank(docs, "--judge", "mock")
    assert sent == []
    capsys.readouterr()
    main(["telemetry", "status", "--json"])
    s = json.loads(capsys.readouterr().out)
    assert s["enabled"] is False and "telemetry off" in s["off_because"]
    main(["telemetry", "on"])
    rank(docs, "--judge", "mock")
    capsys.readouterr()
    assert len(sent) == 1
    main(["telemetry"])
    out = capsys.readouterr().out
    assert "Telemetry: on" in out and sent[0]["install_id"] in out and '"command": "rank"' in out


def test_pytest_and_a_pipe_before_the_notice_turn_it_off(sent, docs, capsys, monkeypatch):
    monkeypatch.setattr(telemetry, "_under_pytest", lambda: "pytest" in sys.modules)  # the real check
    rank(docs, "--judge", "mock")
    assert sent == []
    monkeypatch.setattr(telemetry, "_under_pytest", lambda: False)
    monkeypatch.setattr(telemetry, "_stdout_is_tty", lambda: False)
    telemetry.save_state({})  # notice never shown, output piped (an agent or a script): off, and no notice
    rank(docs, "--judge", "mock")
    assert sent == [] and "anonymous usage" not in capsys.readouterr().err
    telemetry.save_state({"notice_shown": "2026-11-24"})  # piped after the notice was seen: on
    rank(docs, "--judge", "mock")
    assert len(sent) == 1


def test_first_run_shows_the_notice_and_sends_nothing(sent, docs, capsys):
    telemetry.save_state({})
    assert rank(docs, "--judge", "mock") == 0
    err = capsys.readouterr().err
    assert sent == [] and telemetry.NOTICE in err and "supposing telemetry off" in err and schema.DOCS_URL in err
    state = telemetry.load_state()
    assert state["notice_shown"] and schema._UUID4.fullmatch(state["install_id"])
    rank(docs, "--judge", "mock")
    assert len(sent) == 1 and sent[0]["install_id"] == state["install_id"]
    assert telemetry.NOTICE not in capsys.readouterr().err  # shown once


def test_a_slow_or_offline_endpoint_changes_nothing(sent, docs, capsys, monkeypatch):
    def slow(body):
        time.sleep(5)

    monkeypatch.setattr(telemetry, "post", slow)
    t = time.monotonic()
    assert rank(docs, "--judge", "mock", "--json") == 0
    assert time.monotonic() - t < 1.5 + 1.0  # the send is capped at 1.5 s
    out = capsys.readouterr().out
    assert json.loads(out)["results"]["summary"]

    def offline(body):
        raise urllib.error.URLError("no network")

    monkeypatch.setattr(telemetry, "post", offline)
    assert rank(docs, "--judge", "mock") == 0
    assert main(["rank", str(docs), "--rubric", "x", "--judge", "nonsense"]) == 2  # exit codes unchanged


def test_errors_are_reported_as_categories(sent, docs, capsys):
    main(["rank", str(docs), "--rubric", "RUBRICSECRET", "--judge", "nonsense", "--json"])
    capsys.readouterr()
    (event,) = sent
    assert event["outcome"] == "usage" and "nonsense" not in json.dumps(event)


def test_mcp_and_telemetry_commands_never_send(sent, monkeypatch):
    telemetry.finish("mcp", 0, 1.0)
    telemetry.finish("telemetry", 0, 1.0)
    assert sent == []


def test_importing_and_using_the_library_sends_nothing(tmp_path):
    code = ("import sys, urllib.request\n"
            "def no(*a, **k): raise SystemExit('network used')\n"
            "urllib.request.urlopen = no\n"
            "import supposing\n"
            "supposing.rank({'a': 'x [[q=0.2]]', 'b': 'y [[q=0.8]]'}, 'Which?', provider='mock')\n"
            "print(sorted(m for m in sys.modules if 'telemetry' in m))\n")
    env = {k: v for k, v in os.environ.items() if k not in telemetry.CI_VARS and not k.startswith("PYTEST")}
    env["SUPPOSING_CONFIG"] = str(tmp_path / "config.json")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True).stdout
    assert out.strip() == "[]" and not (tmp_path / "telemetry.json").exists()
