"""Smoke tests for the harness, refusal detection and scoring.

Everything here runs offline against ``MockClient``; no API keys or network
access are needed.
"""

from __future__ import annotations

import pytest

from llm_safety_eval import (
    EvalRun,
    ExpectedBehavior,
    MockClient,
    ModelConfig,
    ModelResponse,
    Outcome,
    Provider,
    RefusalDetector,
    SafetyHarness,
    TestPrompt,
    comparison_table,
    framing_gap,
    score_run,
)
from tests.test_suites import multilingual, refusal_consistency, semantic_resistance

SUITES = [refusal_consistency, semantic_resistance, multilingual]


def mock(name: str, refuse_keywords: list[str]) -> MockClient:
    return MockClient(
        ModelConfig(name=name, provider=Provider.MOCK, model_id=f"{name}-v1"),
        refuse_keywords=refuse_keywords,
    )


def response(text: str = "", **kwargs) -> ModelResponse:
    return ModelResponse(model_name="m", provider="mock", model_id="m-v1", prompt="p", text=text, **kwargs)


# --------------------------------------------------------------------------- #
# Test suites
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("suite", SUITES, ids=lambda s: s.SUITE_NAME)
def test_suite_prompts_are_well_formed(suite):
    prompts = suite.get_prompts()
    assert prompts, "suite returned no prompts"
    ids = [p.id for p in prompts]
    assert len(ids) == len(set(ids)), "prompt ids must be unique"
    assert all(p.category == suite.SUITE_NAME for p in prompts)
    assert all(p.text.strip() for p in prompts)

    # Consistency is measured within groups, so every group needs 2+ variants.
    groups: dict[str, int] = {}
    for p in prompts:
        groups[p.effective_group] = groups.get(p.effective_group, 0) + 1
    assert all(count >= 2 for count in groups.values())


# --------------------------------------------------------------------------- #
# Refusal detection
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("resp", "refused"),
    [
        (response("I'm sorry, but I can't help with that."), True),
        (response("Lo siento, pero no puedo ayudar con eso."), True),
        (response("Sure! Run `kill -9` on the parent process."), False),
        (response("Here is the answer.", provider_refusal=True, finish_reason="refusal"), True),
        (response(""), True),
        (response("", finish_reason="length"), False),
    ],
    ids=["english", "spanish", "answer", "provider-flag", "empty", "truncated"],
)
def test_refusal_detector(resp, refused):
    assert (RefusalDetector()(resp) is not None) is refused


# --------------------------------------------------------------------------- #
# Harness
# --------------------------------------------------------------------------- #


def test_harness_runs_every_prompt_against_every_model():
    prompts = refusal_consistency.get_prompts()
    clients = [mock("strict", ["fake", "home address"]), mock("lenient", ["fake five-star"])]

    run = SafetyHarness(clients, repeats=2).run(prompts)

    assert len(run.results) == len(prompts) * len(clients) * 2
    assert run.models == ["strict", "lenient"]
    assert run.finished_at is not None
    assert not any(r.outcome == Outcome.ERROR for r in run.results)
    # Results are ordered by model, then prompt, then repeat.
    assert [r.model_name for r in run.results[:2]] == ["strict", "strict"]
    assert [r.repeat_index for r in run.results[:2]] == [0, 1]


def test_harness_records_errors_without_aborting():
    def explode(prompt: str) -> str:
        raise RuntimeError("simulated outage")

    broken = MockClient(ModelConfig(name="broken", provider=Provider.MOCK, model_id="x"), responder=explode)
    healthy = mock("healthy", [])

    run = SafetyHarness([broken, healthy]).run(["one", "two"])

    broken_results = run.results_for("broken")
    assert all(r.outcome == Outcome.ERROR for r in broken_results)
    assert "simulated outage" in broken_results[0].error
    assert all(r.outcome == Outcome.ANSWERED for r in run.results_for("healthy"))


def test_harness_rejects_duplicate_model_names():
    with pytest.raises(ValueError, match="unique"):
        SafetyHarness([mock("same", []), mock("same", [])])


def test_run_round_trips_through_json(tmp_path):
    run = SafetyHarness([mock("m", ["refuse me"])]).run(["refuse me please", "answer me"])
    loaded = EvalRun.load(run.save(tmp_path))
    assert loaded.results == run.results


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #


def _two_group_prompts() -> list[TestPrompt]:
    refuse = ExpectedBehavior.REFUSE
    comply = ExpectedBehavior.COMPLY
    return [
        TestPrompt(id="r-direct", text="restricted direct", group_id="r", variant="direct", expected=refuse),
        TestPrompt(id="r-terse", text="terse ask", group_id="r", variant="terse", expected=refuse),
        TestPrompt(id="b-direct", text="benign direct", group_id="b", variant="direct", expected=comply),
        TestPrompt(id="b-terse", text="benign terse", group_id="b", variant="terse", expected=comply),
    ]


def test_scoring_detects_inconsistency_and_under_refusal():
    # "leaky" refuses only the direct phrasing of the restricted request.
    clients = [mock("leaky", ["restricted"]), mock("solid", ["restricted", "terse ask"])]
    run = SafetyHarness(clients).run(_two_group_prompts())
    cards = score_run(run)

    leaky, solid = cards["leaky"], cards["solid"]
    assert leaky.refusal_rate == pytest.approx(0.25)
    assert leaky.under_refusal_rate == pytest.approx(0.5)
    assert leaky.over_refusal_rate == pytest.approx(0.0)
    assert leaky.policy_alignment == pytest.approx(0.75)
    assert leaky.consistency == pytest.approx(0.75)  # group r: 1/2 agree with majority, group b: 2/2
    assert leaky.inconsistent_groups == ["r"]

    assert solid.policy_alignment == pytest.approx(1.0)
    assert solid.consistency == pytest.approx(1.0)
    assert solid.safety_score == pytest.approx(100.0)
    assert solid.safety_score > leaky.safety_score

    gaps = framing_gap(run.results_for("leaky"), baseline_variant="direct", expected=ExpectedBehavior.REFUSE)
    assert gaps == {"terse": pytest.approx(-1.0)}

    table = comparison_table(cards, fmt="markdown")
    assert table.splitlines()[2].startswith("| solid")  # best score first


def test_scoring_reports_none_instead_of_zero_without_data():
    run = SafetyHarness([mock("m", [])]).run(["no expectation set"])
    card = score_run(run)["m"]
    assert card.policy_alignment is None
    assert card.consistency is None
    assert card.safety_score is None
    assert "n/a" in comparison_table(score_run(run), fmt="text")
