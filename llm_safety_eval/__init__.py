"""llm-safety-eval: test whether a model's behavior matches its stated safety policy.

Quick start::

    from llm_safety_eval import SafetyHarness, create_clients, load_config, score_run, comparison_table

    config = load_config()
    harness = SafetyHarness(create_clients(config.available_models()))
    run = harness.run(["What are common signs of a phishing email?"])
    print(comparison_table(score_run(run)))
"""

from .config import EvalConfig, ModelConfig, Provider, load_config
from .harness import (
    DEFAULT_REFUSAL_PATTERNS,
    EvalRun,
    ExpectedBehavior,
    Outcome,
    RefusalDetector,
    SafetyHarness,
    TestPrompt,
    TestResult,
)
from .models import (
    AnthropicClient,
    DeepSeekClient,
    GeminiClient,
    LLMClient,
    LocalOpenAICompatibleClient,
    MockClient,
    ModelResponse,
    OpenAIClient,
    create_client,
    create_clients,
)
from .scoring import (
    ModelScorecard,
    ScoreWeights,
    category_table,
    comparison_table,
    consistency_score,
    framing_gap,
    policy_alignment,
    refusal_rate,
    refusal_rate_by,
    score_model,
    score_run,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # config
    "EvalConfig",
    "ModelConfig",
    "Provider",
    "load_config",
    # harness
    "DEFAULT_REFUSAL_PATTERNS",
    "EvalRun",
    "ExpectedBehavior",
    "Outcome",
    "RefusalDetector",
    "SafetyHarness",
    "TestPrompt",
    "TestResult",
    # models
    "AnthropicClient",
    "DeepSeekClient",
    "GeminiClient",
    "LLMClient",
    "LocalOpenAICompatibleClient",
    "MockClient",
    "ModelResponse",
    "OpenAIClient",
    "create_client",
    "create_clients",
    # scoring
    "ModelScorecard",
    "ScoreWeights",
    "category_table",
    "comparison_table",
    "consistency_score",
    "framing_gap",
    "policy_alignment",
    "refusal_rate",
    "refusal_rate_by",
    "score_model",
    "score_run",
]
