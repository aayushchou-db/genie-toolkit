import logging
import random

import gepa.optimize_anything as oa
import yaml
from gepa.optimize_anything import EngineConfig, GEPAConfig, optimize_anything

from genie_toolkit.genie_service import GenieService
from genie_toolkit.schemas import (
    GenieBenchmarkQuestion,
    GenieInstructions,
    GenieSchemaSettings,
)

logger = logging.getLogger(__name__)

SEED = 42


def split_benchmarks(
    questions: list[GenieBenchmarkQuestion],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> tuple[
    list[GenieBenchmarkQuestion],
    list[GenieBenchmarkQuestion],
    list[GenieBenchmarkQuestion],
]:
    """Shuffle and split benchmark questions into train/val/test sets."""
    if train_ratio + val_ratio > 1.0:
        raise ValueError("train_ratio + val_ratio must be <= 1.0")
    if len(questions) < 3:
        raise ValueError(
            "Need at least 3 benchmark questions to split into train/val/test"
        )

    items = list(questions)
    rng = random.Random(SEED)
    rng.shuffle(items)

    n = len(items)
    train_end = max(1, int(n * train_ratio))
    val_end = max(train_end + 1, int(n * (train_ratio + val_ratio)))

    return items[:train_end], items[train_end:val_end], items[val_end:]


def extract_response_text(message) -> str:
    """Extract text and/or SQL from a GenieMessage's attachments."""
    parts = []
    if message.attachments:
        for att in message.attachments:
            if att.text and att.text.content:
                parts.append(att.text.content)
            if att.query and att.query.query:
                parts.append(att.query.query)
    return "\n".join(parts) if parts else ""


def query_genie_space(genie_service: GenieService, space_id: str, question: str) -> str:
    """Send a question to a Genie Space and return the response text/SQL."""
    try:
        message = genie_service.ask_question(space_id, question)
        return extract_response_text(message)
    except Exception as e:
        logger.warning(f"Genie query failed for {question!r}: {e}")
        return ""


def llm_judge(
    wc,
    question: str,
    expected_answer: str,
    actual_answer: str,
    model_endpoint: str,
) -> tuple[int, str]:
    """Use a Databricks Foundation Model to judge answer equivalence.

    Returns (score, diagnostic_text) where score is 0 or 1.
    """
    prompt = (
        "You are an evaluation judge. Determine whether the actual answer is "
        "semantically equivalent to the expected answer for the given question.\n\n"
        f"Question: {question}\n"
        f"Expected answer: {expected_answer}\n"
        f"Actual answer: {actual_answer}\n\n"
        "Respond with ONLY a JSON object: "
        '{"score": 1, "reason": "..."} if equivalent, '
        '{"score": 0, "reason": "..."} if not.'
    )

    try:
        response = wc.serving_endpoints.query(
            name=model_endpoint,
            messages=[{"role": "user", "content": prompt}],
        )
        content = response.choices[0].message.content
        import json

        parsed = json.loads(content)
        score = int(parsed.get("score", 0))
        reason = parsed.get("reason", "")
        return (min(max(score, 0), 1), reason)
    except Exception as e:
        diagnostic = f"LLM judge error: {e}"
        logger.warning(diagnostic)
        return (0, diagnostic)


def serialise_config_components(
    instructions: GenieInstructions | None,
    config: dict | None = None,
) -> str:
    """Convert optimisable config parts into a YAML string for GEPA to mutate."""
    components: dict = {}

    if instructions:
        if instructions.text_instructions:
            components["text_instructions"] = [
                {"id": ti.id, "content": ti.content}
                for ti in instructions.text_instructions
            ]
        if instructions.example_question_sqls:
            components["example_question_sqls"] = [
                {"id": eq.id, "question": eq.question, "sql": eq.sql}
                for eq in instructions.example_question_sqls
            ]
        if instructions.join_specs:
            components["join_specs"] = [
                js.model_dump(exclude_none=True) for js in instructions.join_specs
            ]
        if instructions.sql_snippets:
            components["sql_snippets"] = instructions.sql_snippets.model_dump(
                exclude_none=True
            )

    if config:
        components["sample_questions"] = config

    return yaml.dump(components, sort_keys=False, default_flow_style=False)


def deserialise_config_components(
    candidate_text: str,
) -> tuple[GenieInstructions, dict | None]:
    """Parse a GEPA candidate string back into instruction/config models."""
    data = yaml.safe_load(candidate_text) or {}

    instructions = GenieInstructions(
        text_instructions=data.get("text_instructions"),
        example_question_sqls=data.get("example_question_sqls"),
        join_specs=data.get("join_specs"),
        sql_snippets=data.get("sql_snippets"),
    )

    sample_questions = data.get("sample_questions")
    return instructions, sample_questions


def _format_expected_answer(benchmark_q: GenieBenchmarkQuestion) -> str:
    """Flatten benchmark answer into a single string for comparison."""
    parts = []
    for ans in benchmark_q.answer:
        if ans.content:
            parts.extend(ans.content)
    return "\n".join(parts)


def _evaluate_questions(
    genie_service: GenieService,
    space_id: str,
    questions: list[GenieBenchmarkQuestion],
    wc,
    model_endpoint: str,
) -> float:
    """Run questions against Genie and LLM-judge them. Returns aggregate score."""
    if not questions:
        return 0.0

    total = 0
    for bq in questions:
        question_text = " ".join(bq.question)
        actual = query_genie_space(genie_service, space_id, question_text)
        expected = _format_expected_answer(bq)
        score, reason = llm_judge(wc, question_text, expected, actual, model_endpoint)
        oa.log(f"Q: {question_text} | Score: {score} | Reason: {reason}")
        total += score

    return total / len(questions)


def build_evaluator(
    genie_service: GenieService,
    space_id: str,
    benchmark_questions: list[GenieBenchmarkQuestion],
    wc,
    model_endpoint: str,
    settings: GenieSchemaSettings,
):
    """Return a closure that GEPA calls as its evaluator.

    Each call: parse candidate → update space config → run questions → score.
    """

    def evaluator(candidate: str) -> float:
        try:
            instructions, sample_questions = deserialise_config_components(candidate)
        except Exception as e:
            oa.log(f"Failed to parse candidate: {e}")
            return 0.0

        updated_settings = GenieSchemaSettings(
            version=settings.version,
            config=settings.config,
            data_sources=settings.data_sources,
            instructions=instructions,
            benchmarks=settings.benchmarks,
        )

        if sample_questions and settings.config:
            updated_settings.config.sample_questions = sample_questions

        try:
            genie_service.update(space_id, updated_settings)
        except Exception as e:
            oa.log(f"Failed to update Genie Space: {e}")
            return 0.0

        return _evaluate_questions(
            genie_service, space_id, benchmark_questions, wc, model_endpoint
        )

    return evaluator


def run_optimisation(
    genie_service: GenieService,
    space_id: str,
    settings: GenieSchemaSettings,
    benchmarks_questions: list[GenieBenchmarkQuestion],
    model_endpoint: str,
    max_evals: int = 100,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> dict:
    """Orchestrate the full optimisation pipeline.

    Returns a dict with train_score, val_score, test_count, and best_candidate.
    """
    wc = genie_service.wc

    train, val, test = split_benchmarks(benchmarks_questions, train_ratio, val_ratio)

    seed = serialise_config_components(
        settings.instructions,
        settings.config.model_dump(exclude_none=True) if settings.config else None,
    )

    evaluator = build_evaluator(
        genie_service, space_id, train, wc, model_endpoint, settings
    )

    result = optimize_anything(
        seed_candidate=seed,
        evaluator=evaluator,
        objective=(
            "Optimise the Genie Space configuration so that the Genie produces "
            "correct answers to the benchmark questions. Improve text instructions, "
            "example SQL, sample questions, join specs, and SQL snippets."
        ),
        config=GEPAConfig(engine=EngineConfig(max_metric_calls=max_evals)),
    )

    best_candidate = result.best_candidate

    # Apply best candidate and evaluate on validation set
    instructions, sample_questions = deserialise_config_components(best_candidate)
    best_settings = GenieSchemaSettings(
        version=settings.version,
        config=settings.config,
        data_sources=settings.data_sources,
        instructions=instructions,
        benchmarks=settings.benchmarks,
    )
    if sample_questions and best_settings.config:
        best_settings.config.sample_questions = sample_questions

    genie_service.update(space_id, best_settings)

    val_score = _evaluate_questions(genie_service, space_id, val, wc, model_endpoint)

    return {
        "train_score": result.best_score,
        "val_score": val_score,
        "test_count": len(test),
        "best_candidate": best_candidate,
        "best_settings": best_settings,
    }
