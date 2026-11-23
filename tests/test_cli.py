from __future__ import annotations

import io
import json
import stat
from functools import partial

import pytest
from fake_api import KEY, FakeAPI

from supposing.cli import main
from supposing.cloud import Client, config_path


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("SUPPOSING_CONFIG", str(tmp_path / "config.json"))
    monkeypatch.delenv("SUPPOSING_API_KEY", raising=False)
    monkeypatch.setenv("SUPPOSING_API_URL", "http://fake")
    monkeypatch.setattr("supposing.cloud.client.time.sleep", lambda s: None)
    monkeypatch.setattr("sys.stdin", io.StringIO(""))  # not a terminal: never prompt


@pytest.fixture
def api(monkeypatch):
    fake = FakeAPI(polls_until_done=2)
    fake_client = partial(Client, transport=fake.transport())
    for where in ("supposing.cli.cloud.Client", "supposing.cli.billing.Client", "supposing.cloud.Client"):
        monkeypatch.setattr(where, fake_client)
    return fake


@pytest.fixture
def docs(tmp_path):
    d = tmp_path / "docs"
    d.mkdir()
    for i, q in enumerate([0.9, 0.5, 0.1]):
        (d / ("doc%d.md" % i)).write_text("Document %d [[q=%s]]" % (i, q))
    return d


def run_cli(capsys, *argv: str) -> tuple[int, str, str]:
    code = main(list(argv))
    out = capsys.readouterr()
    return code, out.out, out.err


def test_login_saves_key_mode_600(api, capsys, monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO(KEY + "\n"))
    code, out, _ = run_cli(capsys, "login", "--json")
    assert code == 0 and json.loads(out)["org_name"] == "Test"
    assert stat.S_IMODE(config_path().stat().st_mode) == 0o600
    assert KEY not in out  # the key is never printed
    code, out, _ = run_cli(capsys, "whoami", "--json")
    assert code == 0 and json.loads(out)["key_from"] == str(config_path())
    assert run_cli(capsys, "logout")[0] == 0 and not config_path().exists()


def test_login_rejects_bad_key(api, capsys):
    code, out, _ = run_cli(capsys, "login", "--api-key", "sp_wrong", "--json")
    assert code == 3 and json.loads(out)["error"]["code"] == "unauthorized"
    assert not config_path().exists()


def test_no_key_says_what_to_do(api, capsys):
    code, out, err = run_cli(capsys, "datasets", "list")
    assert code == 3 and "supposing login" in err and out == ""


def test_run_from_folder_end_to_end(api, docs, capsys, monkeypatch):
    monkeypatch.setenv("SUPPOSING_API_KEY", KEY)
    code, out, err = run_cli(capsys, "run", str(docs), "--rubric", "Better?", "--provider", "mock", "--rounds", "2",
                             "--json")
    assert code == 0, err
    doc = json.loads(out)
    assert set(doc) == {"run", "estimate", "results", "skipped"} and doc["skipped"] == []
    assert doc["estimate"]["est_cost_usd"] == 0
    assert doc["run"]["status"] == "done" and doc["run"]["ranking"] == {"rubric": "Better?", "rounds": 2}
    assert doc["results"]["summary"] and doc["results"]["ranking"]["leaderboard"][0]["title"] == "doc0.md"
    sent_req = next(r for r in api.requests if r.url.path == "/v1/runs")
    sent = json.loads(sent_req.content)
    assert sent["judge"] == {"provider": "mock", "model": "mock"} and sent["type"] == "ranking"
    assert sent_req.headers["supposing-source"] == "cli"

    code, out, _ = run_cli(capsys, "results", doc["run"]["id"])
    assert code == 0 and out.startswith("doc0.md ranks first.") and "could be" in out


def test_run_and_rank_print_the_same_shape(api, docs, capsys, monkeypatch):
    monkeypatch.setenv("SUPPOSING_API_KEY", KEY)
    flags = ["--rubric", "Better?", "--provider", "mock", "--rounds", "2", "--json"]
    _, cloud_out, _ = run_cli(capsys, "run", str(docs), *flags)
    code, local_out, _ = run_cli(capsys, "rank", str(docs), *flags)
    assert code == 0
    cloud_doc, local_doc = json.loads(cloud_out), json.loads(local_out)
    assert set(cloud_doc) == set(local_doc)
    assert set(cloud_doc["results"]) == set(local_doc["results"])
    assert set(cloud_doc["results"]["ranking"]) == set(local_doc["results"]["ranking"])
    entry = set(local_doc["results"]["ranking"]["leaderboard"][0])
    assert {"item_id", "title", "rank", "rank_lo", "rank_hi", "rating", "rating_lo", "rating_hi"} <= entry
    assert local_doc["results"]["ranking"]["leaderboard"][0]["title"] == "doc0.md"


def test_paid_run_needs_yes(api, docs, capsys, monkeypatch):
    monkeypatch.setenv("SUPPOSING_API_KEY", KEY)
    real = api.handle

    def priced(req):
        resp = real(req)
        if req.url.path == "/v1/runs/estimate":
            body = {**resp.json(), "est_cost_usd": 0.42, "est_cost_usd_min": 0.1, "billed": False}  # own key
            return type(resp)(200, json=body)
        return resp
    api.handle = priced
    code, out, _ = run_cli(capsys, "run", str(docs), "--rubric", "Better?", "--json")
    err = json.loads(out)["error"]
    assert code == 7 and err["code"] == "confirmation_required" and "--yes" in err["hint"]
    assert not api.runs  # nothing started
    code, _, stderr = run_cli(capsys, "run", str(docs), "--rubric", "Better?", "--yes")
    assert code == 0 and "$0.10 to $0.42" in stderr and len(api.runs) == 1


def test_dry_run_starts_nothing(api, docs, capsys, monkeypatch):
    monkeypatch.setenv("SUPPOSING_API_KEY", KEY)
    code, out, _ = run_cli(capsys, "run", str(docs), "--rubric", "Better?", "--dry-run", "--json")
    assert code == 0 and json.loads(out)["estimate"]["max_calls"] == 12 and not api.runs


def test_detach_then_watch_and_cancel(api, docs, capsys, monkeypatch):
    monkeypatch.setenv("SUPPOSING_API_KEY", KEY)
    code, out, _ = run_cli(capsys, "run", str(docs), "--rubric", "Better?", "--provider", "mock", "--detach")
    run_id = out.strip()
    assert code == 0 and run_id in api.runs
    code, out, _ = run_cli(capsys, "runs", "watch", run_id, "--json")
    assert code == 0 and json.loads(out)["run"]["status"] == "done"
    _, out, _ = run_cli(capsys, "run", str(docs), "--rubric", "x", "--provider", "mock", "--detach")
    other = out.strip()
    assert run_cli(capsys, "runs", "cancel", other)[0] == 0
    code, out, _ = run_cli(capsys, "runs", "watch", other, "--json")
    assert code == 1 and json.loads(out)["run"]["status"] == "canceled"


def test_list_show_and_not_found(api, docs, capsys, monkeypatch):
    monkeypatch.setenv("SUPPOSING_API_KEY", KEY)
    code, out, _ = run_cli(capsys, "datasets", "push", str(docs))
    ds_id = out.strip()
    assert code == 0 and ds_id in api.datasets
    code, out, _ = run_cli(capsys, "datasets", "list")
    assert code == 0 and ds_id in out
    code, out, _ = run_cli(capsys, "datasets", "get", ds_id, "--json")
    assert json.loads(out)["n_items"] == 3
    code, out, _ = run_cli(capsys, "datasets", "get", "00000000-0000-0000-0000-000000000000", "--json")
    assert code == 4 and json.loads(out)["error"]["code"] == "dataset_not_found"
    assert run_cli(capsys, "runs", "list", "--json")[0] == 0
    code, _, err = run_cli(capsys, "run", "not-a-folder", "--rubric", "x")
    assert code == 2 and "neither a folder" in err and "nor a dataset id" in err


def test_rank_needs_two_documents(tmp_path, capsys):
    (tmp_path / "only.md").write_text("one")
    code, out, _ = run_cli(capsys, "rank", str(tmp_path), "--rubric", "x", "--provider", "mock", "--json")
    assert code == 2 and json.loads(out)["error"]["code"] == "too_few_items"


def test_rubric_from_file(tmp_path, docs, capsys):
    (tmp_path / "rubric.md").write_text("Prefer the clearest.\n")
    code, out, _ = run_cli(capsys, "rank", str(docs), "--rubric", "@%s" % (tmp_path / "rubric.md"),
                           "--provider", "mock", "--rounds", "1", "--json")
    assert code == 0 and json.loads(out)["run"]["ranking"]["rubric"] == "Prefer the clearest."


def test_every_command_takes_json():
    from supposing.cli import build_parser

    parser = build_parser()
    seen = []

    def walk(p, path):
        subs = [a for a in p._actions if a.__class__.__name__ == "_SubParsersAction"]
        if not subs:
            seen.append(path)
            assert any("--json" in a.option_strings for a in p._actions), path
        for s in subs:
            for name, child in s.choices.items():
                walk(child, [*path, name])

    walk(parser, [])
    assert len(seen) >= 15


def test_mcp_without_the_extra_says_how_to_install(capsys, monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "supposing.mcp_server", None)  # importing it now raises ImportError
    code, out, _ = run_cli(capsys, "mcp", "--json")
    err = json.loads(out)["error"]
    assert code == 2 and err["code"] == "mcp_not_installed" and "supposing[mcp]" in err["hint"]


def test_mcp_passes_mode_and_args(capsys, monkeypatch):
    import sys
    import types

    seen = []
    fake = types.ModuleType("supposing.mcp_server")
    fake.main = lambda argv: seen.append(argv) or 0  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "supposing.mcp_server", fake)
    assert run_cli(capsys, "mcp", "--local")[0] == 0
    assert seen == [["--local"]]


def test_criteria_is_not_a_flag(tmp_path):
    with pytest.raises(SystemExit) as e:
        main(["rank", str(tmp_path), "--criteria", "x"])
    assert e.value.code == 2


def test_retrying_a_run_reuses_the_upload(api, docs, capsys, monkeypatch):
    """An agent that gets exit 7 and retries with --yes must not leave a second copy of the folder behind."""
    monkeypatch.setenv("SUPPOSING_API_KEY", KEY)
    assert run_cli(capsys, "run", str(docs), "--rubric", "x", "--provider", "mock", "--detach")[0] == 0
    code, _, err = run_cli(capsys, "run", str(docs), "--rubric", "x", "--provider", "mock", "--detach")
    assert code == 0 and "uploaded before" in err and len(api.datasets) == 1
    (docs / "doc0.md").write_text("Changed text")  # different files: a new dataset
    run_cli(capsys, "run", str(docs), "--rubric", "x", "--provider", "mock", "--detach")
    assert len(api.datasets) == 2


def test_unavailable_provider_is_refused_before_uploading(api, docs, capsys, monkeypatch):
    monkeypatch.setenv("SUPPOSING_API_KEY", KEY)
    api.providers = ["mock"]  # a workspace without an OpenRouter key
    for flags in (["--provider", "openrouter"], []):  # no --provider means the API's default, openrouter
        code, out, _ = run_cli(capsys, "run", str(docs), "--rubric", "x", *flags, "--json")
        err = json.loads(out)["error"]
        assert code == 5 and err["code"] == "provider_unavailable" and "mock" in err["hint"]
    assert not api.datasets


def test_rank_without_a_key_fails_before_asking_to_spend(docs, capsys, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    code, out, _ = run_cli(capsys, "rank", str(docs), "--rubric", "x", "--json")
    err = json.loads(out)["error"]
    assert code == 2 and err["code"] == "provider_unavailable" and "OPENROUTER_API_KEY" in err["hint"]


def test_usage_errors_are_json_with_json(capsys):
    code, out, err = run_cli(capsys, "run", "./x", "--provider", "mock", "--json")
    body = json.loads(out)["error"]
    assert code == 2 and body["code"] == "usage" and "--rubric" in body["message"] and err == ""
    with pytest.raises(SystemExit):
        main(["run", "./x"])  # without --json, argparse's usage text as usual
    assert "required" in capsys.readouterr().err


def test_rank_prints_the_engines_judge_line(docs, capsys):
    code, out, _ = run_cli(capsys, "rank", str(docs), "--rubric", "x", "--provider", "mock", "--rounds", "2",
                           "--json")
    line = json.loads(out)["results"]["ranking"]["diagnostics"]["reliability"]["line"]
    _, text, _ = run_cli(capsys, "rank", str(docs), "--rubric", "x", "--provider", "mock", "--rounds", "2")
    assert code == 0 and line in text  # once: in the summary or on its own "Judge:" line
    assert text.count(line) == 1


def test_json_reports_skipped_files(api, docs, capsys, monkeypatch):
    monkeypatch.setenv("SUPPOSING_API_KEY", KEY)
    (docs / "budget.pdf").write_bytes(b"%PDF")
    flags = ["--rubric", "x", "--provider", "mock", "--rounds", "1", "--json"]
    for cmd in (["rank", str(docs)], ["run", str(docs)], ["run", str(docs), "--dry-run"]):
        code, out, err = run_cli(capsys, *cmd, *flags)
        assert code == 0 and json.loads(out)["skipped"] == [{"path": "budget.pdf", "reason": "not .md, .txt or .html"}]
        assert err == ""


def test_rubric_file_keeps_criteria_and_weights(api, docs, tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("SUPPOSING_API_KEY", KEY)
    (tmp_path / "grant.md").write_text("# Grants\n\nWhich proposal is most fundable?\n\n## Criteria\n\n"
                                       "- Impact (weight 2): how many people benefit.\n- Feasibility: can they do it?\n")
    for cmd in ("rank", "run"):
        code, out, _ = run_cli(capsys, cmd, str(docs), "--rubric", "@%s" % (tmp_path / "grant.md"),
                               "--provider", "mock", "--rounds", "1", "--json")
        rubric = json.loads(out)["run"]["ranking"]["rubric"]
        assert code == 0 and "Which proposal is most fundable?" in rubric and "Impact" in rubric, rubric
    sent = json.loads(next(r for r in api.requests if r.url.path == "/v1/runs").content)
    assert "Feasibility" in sent["ranking"]["rubric"]


def test_login_without_a_terminal_or_key_says_how(api, capsys):
    code, out, _ = run_cli(capsys, "login", "--json")  # stdin is empty and not a terminal (fixture)
    err = json.loads(out)["error"]
    assert code == 3 and err["code"] == "no_api_key" and "--api-key" in err["hint"]


def test_missing_rubric_file_says_what_to_do(tmp_path, capsys):
    code, out, _ = run_cli(capsys, "rank", str(tmp_path), "--rubric", "@nope.md", "--json")
    msg = json.loads(out)["error"]["message"]
    assert code == 2 and "nope.md" in msg and "Create it" in msg and "in quotes" in msg


def test_billing_commands(api, capsys, monkeypatch):
    monkeypatch.setenv("SUPPOSING_API_KEY", KEY)
    code, out, _ = run_cli(capsys, "billing", "--json")
    assert code == 0 and json.loads(out)["balance_usd"] == 2.0
    code, out, _ = run_cli(capsys, "billing")
    assert code == 0 and "Balance $2.00" in out and "supposing billing topup <10|25|50>" in out
    code, out, _ = run_cli(capsys, "billing", "ledger", "--json")
    assert code == 0 and json.loads(out)["data"][0]["kind"] == "grant"
    code, out, err = run_cli(capsys, "billing", "topup", "10")
    assert code == 0 and out.strip() == "https://checkout.test/cs_test_1" and "$10.00" in err
    sent = json.loads(next(r for r in api.requests if r.url.path == "/v1/billing/checkout").content)
    assert sent == {"amount_usd": 10}
    code, out, _ = run_cli(capsys, "billing", "topup", "ten", "--json")
    assert code == 2 and "whole US dollars" in json.loads(out)["error"]["message"]
    code, out, _ = run_cli(capsys, "billing", "portal")
    assert code == 0 and out.strip() == "https://portal.test/p"
    code, out, _ = run_cli(capsys, "usage")
    assert code == 0 and "3 runs" in out and "Credit balance $2.00" in out
    code, out, _ = run_cli(capsys, "whoami")
    assert code == 0 and "Credit balance $2.00" in out


def test_pricing_needs_no_key(api, capsys):
    code, out, _ = run_cli(capsys, "pricing")
    assert code == 0 and "(provisional)" in out and "economy" in out and "Claude Haiku 5.5" in out
    assert "24,000 characters" in out
    code, out, _ = run_cli(capsys, "pricing", "--json")
    assert json.loads(out)["tiers"][0]["price_per_call_usd"] == 0.004


def test_billed_estimate_shows_price_and_balance_after(api, docs, capsys, monkeypatch):
    monkeypatch.setenv("SUPPOSING_API_KEY", KEY)
    code, _, err = run_cli(capsys, "run", str(docs), "--rubric", "x", "--dry-run")
    assert code == 0, err
    assert "$0.004 a call (economy tier)" in err and "balance $2.00, $1.95 after at most" in err


def test_not_enough_credit_stops_before_spending(api, docs, capsys, monkeypatch):
    monkeypatch.setenv("SUPPOSING_API_KEY", KEY)
    api.balance = 0.01
    code, out, _ = run_cli(capsys, "run", str(docs), "--rubric", "x", "--yes", "--json")
    err = json.loads(out)["error"]
    assert code == 8 and err["code"] == "insufficient_credit"
    assert err["topup_url"] == "https://app.test/billing" and err["topup_command"] == "supposing billing topup 10"
    assert not api.runs
    code, _, stderr = run_cli(capsys, "run", str(docs), "--rubric", "x", "--yes")
    assert code == 8 and "short by $0.04" in stderr and "supposing billing topup 10" in stderr


def test_api_402_is_shown_with_where_to_top_up(api, docs, capsys, monkeypatch):
    monkeypatch.setenv("SUPPOSING_API_KEY", KEY)
    api.refuse_runs_for_credit = True  # the server's hold refuses it even though the estimate looked affordable
    code, out, _ = run_cli(capsys, "run", str(docs), "--rubric", "x", "--yes", "--json")
    err = json.loads(out)["error"]
    assert code == 8 and err["code"] == "insufficient_credit" and "https://app.test/billing" in err["hint"]
    assert err["topup_command"] == "supposing billing topup 10" and err["balance_usd"] == 2.0
    code, _, stderr = run_cli(capsys, "run", str(docs), "--rubric", "x", "--yes")
    assert code == 8 and "Add credit now: supposing billing topup 10" in stderr
    assert stderr.count("https://app.test/billing") == 1  # the hint names it; no repeat


@pytest.fixture
def outputs_jsonl(tmp_path):
    p = tmp_path / "outputs.jsonl"
    rows = [{"id": "out-%02d" % i, "model": "m%d" % (i % 2), "prompt": "Explain tides.",
             "output": "Answer %d [[q=0.%02d]]" % (i, 10 + 9 * i)} for i in range(1, 6)]
    rows.append({"id": "out-06", "model": "m0", "prompt": "Explain tides.", "output": ""})  # no text: skipped
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return p


def test_version(capsys):
    with pytest.raises(SystemExit) as e:
        main(["--version"])
    assert e.value.code == 0 and capsys.readouterr().out.startswith("supposing ")


def test_rank_a_jsonl_keeps_source_ids(outputs_jsonl, capsys):
    code, out, err = run_cli(capsys, "rank", str(outputs_jsonl), "--rubric", "Clearest?", "--provider", "mock",
                             "--rounds", "2", "--json")
    assert code == 0, err
    doc = json.loads(out)
    board = doc["results"]["ranking"]["leaderboard"]
    assert {e["item_id"] for e in board} == {"out-01", "out-02", "out-03", "out-04", "out-05"}
    assert all(e["title"] == e["item_id"] for e in board)
    assert doc["skipped"] == [{"path": "outputs.jsonl#out-06", "reason": "no text in output"}]


def test_text_fields_join_and_title_and_metadata(tmp_path, capsys, api, monkeypatch):
    monkeypatch.setenv("SUPPOSING_API_KEY", KEY)
    csv_path = tmp_path / "applications.csv"
    csv_path.write_text("Submission ID,Project title,Summary,Plan,Email\n"
                        "A1,Clean wells,Ten wells for 5000 people.,Drill in spring.,a@x.org\n"
                        "A2,Reading club,Weekly club for 40 children.,Start in May.,b@x.org\n")
    flags = ["--text-field", "Summary", "--text-field", "Plan", "--id-field", "Submission ID",
             "--title-field", "Project title", "--metadata-field", "Email"]
    code, out, err = run_cli(capsys, "datasets", "push", str(csv_path), *flags, "--json")
    assert code == 0, out + err
    sent = json.loads(next(r for r in api.requests if r.url.path == "/v1/datasets").content)
    first = sent["items"][0]
    assert first["title"] == "Clean wells" and first["source_id"] == "A1" and first["metadata"] == {"Email": "a@x.org"}
    assert first["text"] == "## Summary\n\nTen wells for 5000 people.\n\n## Plan\n\nDrill in spring."
    assert first["filename"] == "applications.csv#A1" and "a@x.org" not in first["text"]
    code, out, _ = run_cli(capsys, "rank", str(csv_path), "--rubric", "x", "--provider", "mock", "--rounds", "1",
                           *flags, "--json")
    assert code == 0 and {e["item_id"] for e in json.loads(out)["results"]["ranking"]["leaderboard"]} == {"A1", "A2"}


def test_bad_fields_say_which_exist(outputs_jsonl, tmp_path, capsys):
    code, out, _ = run_cli(capsys, "rank", str(outputs_jsonl), "--rubric", "x", "--provider", "mock",
                           "--text-field", "answer", "--json")
    err = json.loads(out)["error"]
    assert code == 2 and err["code"] == "invalid_file" and "id, model, prompt, output" in err["message"]
    plain = tmp_path / "plain.jsonl"
    plain.write_text('{"id": "1", "body_md": "a"}\n{"id": "2", "body_md": "b"}\n')
    code, out, _ = run_cli(capsys, "rank", str(plain), "--rubric", "x", "--provider", "mock", "--json")
    err = json.loads(out)["error"]
    assert code == 2 and "Which field" in err["message"] and "--text-field" in err["hint"]
    dup = tmp_path / "dup.jsonl"
    dup.write_text('{"id": "1", "text": "a"}\n{"id": "1", "text": "b"}\n')
    code, out, _ = run_cli(capsys, "rank", str(dup), "--rubric", "x", "--provider", "mock", "--json")
    assert code == 2 and "repeats the id" in json.loads(out)["error"]["message"]


def test_field_flags_on_a_folder_are_refused(docs, capsys):
    code, out, _ = run_cli(capsys, "rank", str(docs), "--rubric", "x", "--provider", "mock", "--text-field", "x",
                           "--json")
    assert code == 2 and json.loads(out)["error"]["code"] == "usage"
