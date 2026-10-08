"""End-to-end example: compare refusal consistency of DeepSeek and Claude.

What this script demonstrates
-----------------------------
1. Loading configuration (API keys, model IDs) from environment variables or
   a ``.env`` file with ``llm_safety_eval.config.load_config``.
2. Creating two models, DeepSeek and Claude, through the unified client layer
   in ``llm_safety_eval.models``. A model whose API key isn't set is skipped
   with a clear message, and the run continues with the other one.
3. Selecting a small, five-prompt set from the ``refusal_consistency`` suite.
   The set contains two *groups*: the same request phrased several ways.
     * ``rc-fake-reviews``: asking for fabricated reviews to post under
       made-up customer names. This is a low-severity policy conflict
       (deception) that most usage policies disallow, so the expected
       behavior is to refuse. Nothing dangerous comes out even if a model
       complies.
     * ``rc-kill-process``: a benign Linux sysadmin question that only
       *sounds* violent, so the expected behavior is to answer.
   A model is *consistent* when it makes the same decision for every
   phrasing in a group. Refusing the direct ask but answering the terse or
   contextual one (or refusing the benign question only when it is phrased
   tersely) is exactly the kind of gap this framework is built to surface.
4. Running every prompt through every available model with ``SafetyHarness``.
5. Printing a score table, per-prompt outcomes and inconsistent groups with
   ``llm_safety_eval.scoring``, then saving the raw results to
   ``data/results/``.

Usage (from the repository root)::

    export DEEPSEEK_API_KEY=sk-...       # either or both; a .env file works too
    export ANTHROPIC_API_KEY=sk-ant-...
    python examples/basic_run.py

    # No API keys? Run the same pipeline against offline mock models:
    python examples/basic_run.py --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the package and the bundled suites importable when run as a script.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from llm_safety_eval import (  # noqa: E402
    EvalConfig,
    LLMClient,
    MockClient,
    ModelConfig,
    Provider,
    SafetyHarness,
    TestPrompt,
    TestResult,
    comparison_table,
    create_client,
    load_config,
    score_run,
)
from tests.test_suites import refusal_consistency  # noqa: E402

#: The models this example compares, by their name in ``config.py``.
MODEL_NAMES = ["deepseek", "claude"]

#: Five prompts from the refusal-consistency suite: two phrasing groups.
PROMPT_IDS = [
    "rc-fake-reviews-direct",
    "rc-fake-reviews-terse",
    "rc-fake-reviews-contextual",
    "rc-kill-process-direct",
    "rc-kill-process-terse",
]


def load_prompts() -> list[TestPrompt]:
    """Pick the example's five prompts out of the full suite, in order."""
    by_id = {p.id: p for p in refusal_consistency.get_prompts()}
    missing = [pid for pid in PROMPT_IDS if pid not in by_id]
    if missing:
        raise KeyError(f"Prompts not found in {refusal_consistency.SUITE_NAME}: {', '.join(missing)}")
    return [by_id[pid] for pid in PROMPT_IDS]


def build_clients(config: EvalConfig) -> list[LLMClient]:
    """Create DeepSeek and Claude clients, skipping any without an API key."""
    clients: list[LLMClient] = []
    for name in MODEL_NAMES:
        model_config = config.get(name)
        if not model_config.is_available:
            print(f"  - Skipping {name}: {model_config.api_key_env} is not set.")
            continue
        try:
            client = create_client(model_config)
        except ImportError as exc:  # SDK not installed
            print(f"  - Skipping {name}: {exc}")
            continue
        print(f"  + Using {name} ({model_config.model_id})")
        clients.append(client)
    return clients


def build_mock_clients() -> list[LLMClient]:
    """Offline stand-ins with different (imperfect) policies, for --dry-run.

    The "lenient" mock only spots the literal phrase "fake five-star", so it
    refuses the direct ask but misses the terse paraphrase, which shows what
    an inconsistency looks like in the report.
    """
    return [
        MockClient(
            ModelConfig(name="mock-lenient", provider=Provider.MOCK, model_id="mock-lenient-v1"),
            refuse_keywords=["fake five-star"],
        ),
        MockClient(
            ModelConfig(name="mock-strict", provider=Provider.MOCK, model_id="mock-strict-v1"),
            refuse_keywords=["fake"],
        ),
    ]


def print_progress(result: TestResult, done: int, total: int) -> None:
    print(f"  [{done}/{total}] {result.model_name:<14} {result.prompt_id:<28} -> {result.outcome.value}")


def print_outcomes(results: list[TestResult], models: list[str]) -> None:
    """Print a prompt-by-model grid of outcomes."""
    width = max(len(pid) for pid in PROMPT_IDS)
    print("  " + "Prompt".ljust(width) + "  Expected  " + "  ".join(m.ljust(14) for m in models))
    for pid in PROMPT_IDS:
        row = [r for r in results if r.prompt_id == pid]
        expected = row[0].expected.value if row else ""
        cells = []
        for model in models:
            match = next((r for r in row if r.model_name == model), None)
            cells.append((match.outcome.value if match else "-").ljust(14))
        print("  " + pid.ljust(width) + "  " + expected.ljust(8) + "  " + "  ".join(cells))


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare DeepSeek and Claude on refusal consistency.")
    parser.add_argument("--dry-run", action="store_true", help="Use offline mock models (no API keys needed)")
    args = parser.parse_args()

    print("1. Loading configuration and models")
    config = load_config()
    clients = build_mock_clients() if args.dry_run else build_clients(config)
    if not clients:
        print(
            "\nNo models available. Set DEEPSEEK_API_KEY and/or ANTHROPIC_API_KEY in your "
            "environment or a .env file, or run with --dry-run to try the pipeline offline.",
            file=sys.stderr,
        )
        return 1

    prompts = load_prompts()
    print(f"\n2. Running {len(prompts)} prompts x {len(clients)} model(s)")
    harness = SafetyHarness(clients, progress=print_progress)
    run = harness.run(prompts, metadata={"example": "basic_run", "dry_run": args.dry_run})

    model_names = [c.name for c in clients]
    print("\n3. Outcomes")
    print_outcomes(run.results, model_names)

    scorecards = score_run(run)
    print("\n4. Scores\n")
    print(comparison_table(scorecards, fmt="text"))

    print("\n5. Consistency findings")
    for card in scorecards.values():
        if card.inconsistent_groups:
            groups = ", ".join(card.inconsistent_groups)
            print(f"  ! {card.model_name}: decision changed with phrasing in {groups}")
        else:
            print(f"  = {card.model_name}: same decision for every phrasing")

    errors = [r for r in run.results if r.error]
    for r in errors:
        print(f"  x {r.model_name} errored on {r.prompt_id}: {r.error}")

    results_dir = config.results_dir if config.results_dir.is_absolute() else REPO_ROOT / config.results_dir
    path = run.save(results_dir)
    print(f"\nRaw results (including full responses) saved to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
