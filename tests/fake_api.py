"""An in-memory stand-in for the cloud API, served through httpx.MockTransport. No network."""
from __future__ import annotations

import csv
import json
import re
import uuid
from typing import Any

import httpx

from supposing import display_groups

KEY = "sp_test_key"
PRICE = 0.004  # a stand-in price per call; the real ones come from the API


def _multipart_files(req: httpx.Request) -> list[tuple[str, bytes]]:
    """(filename, content) of every file part in a multipart request."""
    boundary = req.headers["content-type"].split("boundary=")[1].encode()
    out = []
    for part in req.content.split(b"--" + boundary):
        head, _, body = part.partition(b"\r\n\r\n")
        m = re.search(rb'filename="([^"]*)"', head)
        if m:
            out.append((m.group(1).decode(), body[:-2] if body.endswith(b"\r\n") else body))
    return out


def _multipart_field(req: httpx.Request, name: str) -> str | None:
    boundary = req.headers["content-type"].split("boundary=")[1].encode()
    for part in req.content.split(b"--" + boundary):
        head, _, body = part.partition(b"\r\n\r\n")
        if re.search(rb'name="%s"' % name.encode(), head) and b"filename=" not in head:
            return body[:-2].decode() if body.endswith(b"\r\n") else body.decode()
    return None


def _multipart_file(req: httpx.Request) -> str:
    """The text of the one file in a multipart request."""
    body = req.content.decode("utf-8", errors="replace")
    part = body.split("\r\n\r\n", 1)[1]
    return part.rsplit("\r\n--", 1)[0]


def _err(status: int, code: str, message: str, hint: str | None = None) -> httpx.Response:
    return httpx.Response(status, json={"code": code, "message": message, "hint": hint, "details": None})


def _judge(value: Any) -> dict[str, Any]:
    """The API's judge from its short form or object, as a run response carries it."""
    from supposing.cloud import parse_judge

    if isinstance(value, dict):
        return {"provider": "openrouter", "level": None, **value}
    return parse_judge(value)


class FakeAPI:
    def __init__(self, polls_until_done: int = 2):
        self.datasets: dict[str, dict[str, Any]] = {}
        self.runs: dict[str, dict[str, Any]] = {}
        self.requests: list[httpx.Request] = []
        self.polls_until_done = polls_until_done
        self.fail_next: list[int] = []  # status codes to return before serving normally
        self.providers = ["openrouter", "mock"]
        self.billing_enabled = True
        self.imports: dict[str, list[dict[str, str]]] = {}
        self.commits: list[dict[str, Any]] = []
        self.uploads: list[bytes] = []
        self.balance = 2.0  # prepaid credit; runs on "our key" (any provider but mock) are billed at PRICE a call
        self.refuse_runs_for_credit = False  # answer POST /runs with 402 insufficient_credit, as the API's hold does
        self.events = True  # serve GET /runs/{id}/events; False answers 404 like a server without it

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(lambda req: self.handle(req))

    def async_transport(self) -> httpx.MockTransport:
        async def handle(req: httpx.Request) -> httpx.Response:
            return self.handle(req)
        return httpx.MockTransport(handle)

    def handle(self, req: httpx.Request) -> httpx.Response:
        self.requests.append(req)
        if self.fail_next:
            return httpx.Response(self.fail_next.pop(0), text="bad gateway")
        if req.url.path == "/v1/pricing":  # public
            return httpx.Response(200, json={"provisional": True, "unit_chars": 24000, "unit_bytes": 24000, "free_grant_usd": 2.0,
                                             "min_topup_usd": 5, "max_topup_usd": 1000, "tiers": [
                                                 {"name": "economy", "price_per_call_usd": PRICE,
                                                  "examples": ["Claude Haiku 5.5"]},
                                                 {"name": "pro", "price_per_call_usd": 0.06,
                                                  "examples": ["Claude Sonnet 5.5"]}]})
        if req.headers.get("authorization") != "Bearer " + KEY:
            return _err(401, "unauthorized", "Unknown or revoked API key.", "Create a key in Settings.")
        parts = req.url.path.strip("/").split("/")[1:]
        is_json = req.headers.get("content-type", "").startswith("application/json")
        body = json.loads(req.content) if req.content and is_json else None
        m = req.method
        if parts == ["me"]:
            return httpx.Response(200, json={"org_id": str(uuid.uuid4()), "org_name": "Test", "user_id": None,
                                             "email": None, "via": "api_key"})
        if parts == ["billing"]:
            return httpx.Response(200, json={"enabled": self.billing_enabled, "balance_usd": self.balance, "held_usd": 0.0,
                                             "topup_options_usd": [10, 25, 50], "min_topup_usd": 5,
                                             "max_topup_usd": 1000, "free_grant_usd": 2.0,
                                             "topup_url": "https://app.test/billing"})
        if parts == ["billing", "ledger"]:
            return httpx.Response(200, json={"data": [{"id": str(uuid.uuid4()), "kind": "grant", "amount_usd": 2.0,
                                                       "run_id": None, "note": "Welcome credit",
                                                       "created_at": "2026-11-22T10:00:00Z"}], "next_cursor": None})
        if parts == ["billing", "checkout"]:
            return httpx.Response(201, json={"id": "cs_test_1", "url": "https://checkout.test/cs_test_1"})
        if parts == ["billing", "portal"]:
            return httpx.Response(200, json={"url": "https://portal.test/p"})
        if parts == ["usage"]:
            return httpx.Response(200, json={"plan": "free", "month": "2026-11-01", "runs": 3, "judge_calls": 96,
                                             "prompt_tokens": 1, "completion_tokens": 1, "cost_usd": 0.12,
                                             "platform_cost_usd": 0.05, "limits": {}})
        if parts == ["settings"]:
            return httpx.Response(200, json={"has_openrouter_key": False, "platform_key_available": False,
                                             "providers": self.providers})
        if parts == ["imports"] and m == "POST":
            text = _multipart_file(req)
            rows = list(csv.DictReader(text.splitlines()))
            imp = str(uuid.uuid4())
            self.imports[imp] = rows
            cols = list(rows[0]) if rows else []
            guess = {"id_field": cols[0] if cols else None, "title_field": None,
                     "text_fields": [c for c in cols[1:] if c.lower() not in ("email", "name")][-1:],
                     "metadata_fields": [c for c in cols if c.lower() in ("email", "name")]}
            return httpx.Response(201, json={"id": imp, "filename": "f.csv", "format": "csv", "sheets": [],
                                             "sheet": None, "n_rows": len(rows), "columns": [], "rows": [],
                                             "mapping": guess, "warnings": []})
        if len(parts) == 3 and parts[0] == "imports" and parts[2] == "commit":
            self.commits.append(body)
            mp = body["mapping"]
            items, skipped = [], []
            for n, row in enumerate(self.imports[parts[1]], 1):
                parts_ = [(f, row.get(f, "").strip()) for f in mp["text_fields"] if row.get(f, "").strip()]
                if not parts_:
                    skipped.append({"row": n, "reason": "no text"})
                    continue
                text = parts_[0][1] if len(mp["text_fields"]) == 1 else "\n\n".join("## %s\n\n%s" % p for p in parts_)
                sid = row.get(mp["id_field"]) if mp.get("id_field") else str(n)
                items.append({"title": row.get(mp["title_field"]) if mp.get("title_field") else sid, "text": text,
                              "source_id": sid})
            ds_id = str(uuid.uuid4())
            self.datasets[ds_id] = {"id": ds_id, "name": body["name"], "created_at": "2026-11-22T10:00:00Z",
                                    "items": [{"id": str(uuid.uuid4()), **i, "n_words": len(i["text"].split()),
                                               "n_chars": len(i["text"])} for i in items]}
            return httpx.Response(201, json={**self._ds(ds_id), "skipped": skipped, "duplicates": []})
        if parts == ["datasets", "upload"]:  # like the server: title from the first heading, source_id the file
            self.uploads.append(req.content)
            ds_id = str(uuid.uuid4())
            items, skipped, by_text = [], [], {}
            for fname, content in _multipart_files(req):
                base = fname.rsplit("/", 1)[-1]
                text = content.decode("utf-8", errors="replace").strip()
                if fname.lower().endswith(".pdf") or not text:
                    skipped.append({"filename": fname, "reason": "no text"})
                    continue
                heading = next((ln[2:].strip() for ln in text.splitlines() if ln.startswith("# ")), None)
                item = {"id": str(uuid.uuid4()), "title": heading or base.rsplit(".", 1)[0], "text": text,
                        "filename": base, "source_id": base, "n_words": len(text.split()), "n_chars": len(text)}
                items.append(item)
                by_text.setdefault(text, []).append(item)
            self.datasets[ds_id] = {"id": ds_id, "name": _multipart_field(req, "name") or "upload",
                                    "created_at": "2026-11-22T10:00:00Z", "items": items}
            dups = [{"kind": "identical", "similarity": 1.0,
                     "items": [{"item_id": i["id"], "title": i["title"], "source_id": i["source_id"]} for i in g]}
                    for g in by_text.values() if len(g) > 1]
            return httpx.Response(201, json={**self._ds(ds_id), "skipped": skipped, "duplicates": dups})
        if parts == ["datasets"] and m == "POST":
            ds_id = str(uuid.uuid4())
            items = [{"id": str(uuid.uuid4()), "title": i["title"], "text": i["text"], "source_id": i.get("source_id"),
                      "n_words": len(i["text"].split()), "n_chars": len(i["text"])} for i in body["items"]]
            self.datasets[ds_id] = {"id": ds_id, "name": body["name"], "created_at": "2026-11-22T10:00:00Z",
                                    "items": items}
            return httpx.Response(201, json=self._ds(ds_id))
        if parts == ["datasets"]:
            return httpx.Response(200, json={"data": [{**self._ds(d), "items": None} for d in self.datasets],
                                             "next_cursor": None})
        if len(parts) == 2 and parts[0] == "datasets":
            if parts[1] not in self.datasets:
                return _err(404, "dataset_not_found", "Dataset not found.", "List datasets to find its id.")
            return httpx.Response(200, json=self._ds(parts[1]))
        if parts == ["models"]:
            self.models_params = dict(req.url.params)
            lv = [{"id": k, "name": k.title(), "provider": "Anthropic", "judge": _judge(k), "level": k,
                   "recommended": k == "quick", "tier": "economy", "price_per_call_usd": PRICE, "zdr": True,
                   "available": True, "unavailable_reason": None, "action": None,
                   "est_run_usd_min": 0.01 if "dataset_id" in req.url.params else None,
                   "est_run_usd_max": 0.05 if "dataset_id" in req.url.params else None}
                  for k in ("quick", "balanced", "thorough")]
            other = {**lv[0], "id": "openai/gpt-6-luna", "name": "GPT-6 Luna", "level": None, "recommended": False,
                     "zdr": False, "available": False, "unavailable_reason": "tier_not_covered", "action": "own_key"}
            return httpx.Response(200, json={"levels": lv, "default": "quick", "data": [*lv, other], "local": [],
                                             "paid_by": "credit", "balance_usd": self.balance,
                                             "zero_data_retention": False, "next_cursor": None})
        if parts == ["runs", "estimate"]:
            n = len(self.datasets[body["dataset_id"]]["items"])
            billed = _judge(body.get("judge"))["provider"] != "mock"
            return httpx.Response(200, json={"type": "ranking", "n_items": n, "min_calls": n, "max_calls": 4 * n,
                                             "est_input_tokens": 1000, "est_output_tokens": 100, "billed": billed,
                                             "price_tier": "economy" if billed else None,
                                             "price_per_call_usd": PRICE if billed else None,
                                             "est_cost_usd": round(4 * n * PRICE, 4) if billed else 0.0,
                                             "est_cost_usd_min": round(n * PRICE, 4) if billed else 0.0,
                                             "est_seconds": 12})
        if parts == ["runs"] and m == "POST":
            if self.refuse_runs_for_credit:
                return _err(402, "insufficient_credit", "This run needs up to $0.05 of credit and the balance is $0.01.",
                            "Add credit at https://app.test/billing, or use your own OpenRouter key in Settings "
                            "(never billed).")
            if body["dataset_id"] not in self.datasets:
                return _err(404, "dataset_not_found", "Dataset not found.", "List datasets to find its id.")
            run_id = str(uuid.uuid4())
            self.runs[run_id] = {"id": run_id, "type": "ranking", "name": "r", "status": "queued",
                                 "dataset_id": body["dataset_id"], "dataset_name": "d",
                                 "judge": _judge(body.get("judge")),
                                 "ranking": body["ranking"], "max_cost_usd": body.get("max_cost_usd"),
                                 "planned_calls": 8, "note": None, "error": None, "polls": 0,
                                 "created_at": "2026-11-22T10:00:00Z", "started_at": None, "finished_at": None}
            return httpx.Response(201, json=self._run(run_id))
        if parts == ["runs"]:
            return httpx.Response(200, json={"data": [self._run(r) for r in self.runs], "next_cursor": None})
        if len(parts) >= 2 and parts[0] == "runs":
            run = self.runs.get(parts[1])
            if not run:
                return _err(404, "run_not_found", "Run not found.", None)
            if len(parts) == 2 and m == "DELETE":
                del self.runs[parts[1]]
                return httpx.Response(204)
            if len(parts) == 2:
                run["polls"] += 1
                if run["status"] not in ("canceled",):
                    run["status"] = "done" if run["polls"] >= self.polls_until_done else "running"
                return httpx.Response(200, json=self._run(parts[1]))
            if parts[2] == "events":
                if not self.events:
                    return _err(404, "not_found", "No route.")
                lines = [": keep-alive", ""]
                for status in (("canceled",) if run["status"] == "canceled" else ("running", "running", "done")):
                    run["status"] = status
                    leaders = [{"item_id": "x", "title": "doc0.md", "rating": 1500, "rank": 1}]
                    ev = {"event": "progress" if status == "running" else "finished", "run": self._run(parts[1]),
                          "leaders": leaders if status == "done" else []}
                    lines += ["data: " + json.dumps(ev), ""]
                return httpx.Response(200, text="\n".join(lines), headers={"content-type": "text/event-stream"})
            if parts[2] == "cancel":
                run["status"] = "canceled"
                return httpx.Response(200, json=self._run(parts[1]))
            if parts[2] == "results":
                items = self.datasets[run["dataset_id"]]["items"]
                board = [{"item_id": it["id"], "title": it["title"], "source_id": it.get("source_id"),
                          "n_words": it["n_words"], "rating": 1500 - 50 * i,
                          "rating_lo": 1450 - 50 * i, "rating_hi": 1550 - 50 * i, "rank": i + 1, "rank_lo": i + 1,
                          "rank_hi": i + 1, "wins": 3 - i, "losses": i} for i, it in enumerate(items)]
                return httpx.Response(200, json={
                    "run_id": parts[1], "type": "ranking", "final": run["status"] == "done",
                    "summary": "%s ranks first." % items[0]["title"],
                    "ranking": {"leaderboard": board, "diagnostics": {"n_decided": 8}, "interval": "90%",
                                "n_bootstrap": 200, "groups": display_groups(board)}})
            if parts[2] == "comparisons":
                ids = [i["id"] for i in self.datasets[run["dataset_id"]]["items"]][:2]
                titles = [i["title"] for i in self.datasets[run["dataset_id"]]["items"]][:2]
                row = {"id": str(uuid.uuid4()), "run_id": parts[1], "round": 1, "pair_id": 0, "item_a": ids[0],
                       "item_b": ids[1], "title_a": titles[0], "title_b": titles[1], "status": "done",
                       "winner_item": ids[0], "reasoning": "A is clearer.", "error": None, "attempts": 1,
                       "cost_usd": 0.0, "latency_ms": 5, "created_at": "2026-11-22T10:00:00Z"}
                return httpx.Response(200, json={"data": [row], "next_cursor": None})
        return _err(404, "not_found", "No route %s %s" % (m, req.url.path))

    def _ds(self, ds_id: str) -> dict[str, Any]:
        d = self.datasets[ds_id]
        return {**{k: v for k, v in d.items() if k != "items"}, "n_items": len(d["items"]),
                "items": [{k: v for k, v in i.items() if k != "text"} for i in d["items"]]}

    def _run(self, run_id: str) -> dict[str, Any]:
        r = self.runs[run_id]
        done = r["status"] == "done"
        return {**{k: v for k, v in r.items() if k != "polls"},
                "progress": {"total": 8, "done": 8 if done else 3, "failed": 0, "in_flight": 0 if done else 2,
                             "pending": 0 if done else 3, "round": 2, "cost_usd": 0.0}}
