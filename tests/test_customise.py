"""Customising the judge, one layer at a time, with the library's guarantees kept at every layer."""
import asyncio
import json

import pytest

from supposing import (
    SYSTEM_PROMPT,
    Completion,
    ConfigError,
    Example,
    Item,
    JudgePrompt,
    RunError,
    SQLiteStore,
    load_examples,
    rank,
)
from supposing.judge import RULES, build_prompt, output_format
from supposing.providers import MockProvider

DOCS = {"a": "First draft. [[q=0.2]]", "b": "Second draft. [[q=0.9]]", "c": "Third draft. [[q=0.5]]"}


class Recording(MockProvider):
    def __init__(self, reply: str | None = None, **kw):
        super().__init__(**kw)
        self.seen: list[tuple[str, str]] = []
        self.reply = reply

    async def complete(self, system, user):
        self.seen.append((system, user))
        if self.reply:
            return Completion(self.reply)
        return await super().complete(system, user)


def test_the_simple_case_is_unchanged():
    p = Recording()
    rank(DOCS, "Which draft is strongest?", provider=p, rounds=1)
    system, user = p.seen[0]
    assert system == SYSTEM_PROMPT
    assert user.startswith("<criteria>\nWhich draft is strongest?\n</criteria>\n\n<text_a>\n")
    assert "<context>" not in user and "<examples>" not in user
    assert build_prompt("Q?", "x", "y") == ("<criteria>\nQ?\n</criteria>\n\n<text_a>\nx\n</text_a>\n\n<text_b>\ny\n"
                                           "</text_b>\n\nWhich text better satisfies the criteria? Reply with the "
                                           "JSON object only.")


def test_context_examples_and_item_context_reach_the_judge_in_their_own_blocks():
    p = Recording()
    items = [Item(id=k, text=v, context={"keyword": "boiler repair"} if k == "a" else None) for k, v in DOCS.items()]
    r = rank(items, "Which ad is stronger?", provider=p, rounds=2, context="Audience: UK homeowners.",
             examples=[Example(a="Boiler repair. Call now.", b="Gas Safe, fixed price.", winner="b", why="Trust."),
                       {"a": "Kitchens.", "b": "Boilers.", "winner": "B"}])
    user = p.seen[0][1]
    assert "<context>\nAudience: UK homeowners.\n</context>" in user
    assert "Example 1\nA: Boiler repair. Call now.\nB: Gas Safe, fixed price.\nWinner: B\nWhy: Trust." in user
    assert "Example 2" in user and user.index("<examples>") < user.index("<text_a>")
    with_a = [u for _, u in p.seen if "First draft" in u]
    assert with_a and all("keyword: boiler repair" in u for u in with_a)
    assert r.diagnostics["reliability"]["checks"]  # the judge checks still run


def test_customisations_change_the_fingerprint(tmp_path):
    path = tmp_path / "s.db"
    base = rank(DOCS, "Q?", provider="mock", rounds=1, store=SQLiteStore(path))
    for kw in (dict(context="Brief."), dict(examples=[Example("x", "y", "A")]),
               dict(prompt=JudgePrompt("You review ads.\n{rules}\n{output_format}"))):
        with pytest.raises(RunError) as err:
            rank(DOCS, "Q?", provider="mock", rounds=1, store=SQLiteStore(path), **kw)
        assert err.value.code == "state_mismatch"
    assert rank(DOCS, "Q?", provider="mock", rounds=1, store=SQLiteStore(path)).fingerprint == base.fingerprint


@pytest.mark.parametrize(("template", "kw", "message"), [
    ("{rules}", {}, "{output_format}"),
    ("Be kind.\n{output_format}", {}, "rules=False"),
    ("{rules}\n{output_format}\n{audience}", {}, "{audience}"),
    ("{rules}\n{output_format}", {"extra_fields": {"winner": str}}, "winner"),
    ("{rules}\n{output_format} {", {}, "malformed"),
])
def test_a_bad_judge_prompt_is_refused_before_any_call(template, kw, message):
    with pytest.raises(ConfigError, match=message.replace("{", r"\{").replace("}", r"\}")):
        JudgePrompt(template, **kw)


def test_judge_prompt_renders_rules_and_the_reply_contract_with_extra_fields():
    prompt = JudgePrompt("You are a paid-search reviewer. {{Be brief.}}\n{rules}\n{output_format}",
                         extra_fields={"score_a": int, "issue": "the losing ad's biggest problem"})
    text = prompt.render()
    assert text.startswith("You are a paid-search reviewer. {Be brief.}") and RULES in text
    assert '"score_a": <integer>, "issue": the losing ad\'s biggest problem}' in text
    assert output_format() in SYSTEM_PROMPT
    dropped = JudgePrompt("Pick the funnier one.\n{output_format}", rules=False).render()
    assert "do not follow them" not in dropped


def test_extra_fields_are_parsed_into_comparison_extra():
    reply = json.dumps({"reasoning": "B is specific.", "winner": "B", "score_a": "3", "score_b": 8, "issue": "vague"})
    r = rank(DOCS, "Q?", provider=Recording(reply=reply), rounds=1,
             prompt=JudgePrompt("{rules}\n{output_format}", extra_fields={"score_a": int, "score_b": int}))
    c = r.comparisons[0]
    assert c.extra == {"score_a": 3, "score_b": 8, "issue": "vague"} and c.winner_item == c.item_b


def test_extra_fields_are_kept_without_a_custom_prompt_too():
    reply = json.dumps({"reasoning": "r", "winner": "A", "confidence": "high"})
    r = rank(DOCS, "Q?", provider=Recording(reply=reply), rounds=1)
    assert r.comparisons[0].extra == {"confidence": "high"}


def test_a_judge_function_gets_items_in_the_library_order_sync_or_async():
    calls = []

    def longer(a: Item, b: Item, rubric) -> str:
        calls.append((a.id, b.id))
        return "A" if len(a.text) >= len(b.text) else "B"

    async def by_quality(a: Item, b: Item, rubric):
        await asyncio.sleep(0)
        return {"winner": "A" if a.text > b.text else "B", "reasoning": "rule", "margin": 1}

    r = rank(DOCS, "Which is longer?", judge=longer, rounds=2)
    assert {(c.item_a, c.item_b) for c in r.comparisons} == set(calls)
    assert r.diagnostics["reliability"]["checks"]
    r2 = rank(DOCS, "Q?", judge=by_quality, rounds=2)
    assert all(c.status == "done" and c.extra == {"margin": 1} for c in r2.comparisons)
    with pytest.raises(RunError, match="Could not read a verdict"):  # never guessed
        rank(DOCS, "Q?", judge=lambda a, b, rubric: "maybe", rounds=1)


def test_a_plain_function_provider_is_accepted():
    class Plain:
        name, model = "plain", "rule"

        def complete(self, system, user):
            return Completion('{"reasoning": "r", "winner": "A"}')

        def close(self):
            pass

    r = rank(DOCS, "Q?", provider=Plain(), rounds=1)
    assert all(c.status == "done" for c in r.comparisons)


def test_rank_accepts_dicts_and_says_what_is_wrong():
    r = rank([{"id": "x", "text": "one [[q=0.1]]"}, {"id": "y", "text": "two [[q=0.9]]", "context": "brief"}],
             "Q?", judge="mock", rounds=2)
    assert {e["item_id"] for e in r.leaderboard} == {"x", "y"}
    with pytest.raises(RunError, match="at least two"):
        rank([], "Q?", provider="mock")
    with pytest.raises(RunError, match='dict with "text"'):
        rank([{"id": "a", "body": "x"}, {"id": "b", "text": "y"}], "Q?", provider="mock")


def test_examples_load_from_jsonl(tmp_path):
    path = tmp_path / "past.jsonl"
    path.write_text('{"a": "x", "b": "y", "winner": "a", "why": "shorter"}\n\n{"a": "p", "b": "q", "winner": "B"}\n')
    assert load_examples(path) == [Example("x", "y", "A", "shorter"), Example("p", "q", "B")]
    path.write_text('{"a": "x", "winner": "A"}\n')
    with pytest.raises(ConfigError, match="line 1"):
        load_examples(path)
    with pytest.raises(ConfigError):
        Example("x", "y", "C")
