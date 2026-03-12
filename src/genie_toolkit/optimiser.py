import logging
import random

import gepa.optimize_anything as oa
import yaml
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole
from gepa.optimize_anything import (
    EngineConfig,
    GEPAConfig,
    ReflectionConfig,
    optimize_anything,
)

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
    system_prompt = (
        "You are an evaluation judge. Determine whether the actual answer is "
        "semantically equivalent to the expected answer for the given question.\n\n"
        "Respond with ONLY a JSON object: "
        '{"score": 1, "reason": "..."} if equivalent, '
        '{"score": 0, "reason": "..."} if not.'
    )
    user_prompt = (
        f"Question: {question}\n"
        f"Expected answer: {expected_answer}\n"
        f"Actual answer: {actual_answer}\n\n"
    )

    try:
        response = wc.serving_endpoints.query(
            name=model_endpoint,
            messages=[
                ChatMessage(role=ChatMessageRole.SYSTEM, content=system_prompt),
                ChatMessage(role=ChatMessageRole.USER, content=user_prompt),
            ],
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


def _build_background(settings: GenieSchemaSettings) -> str:
    """Build domain context for the GEPA reflection LLM.

    Includes table metadata, structured artifact format examples, and priority guidance.
    """
    sections: list[str] = []

    # Table metadata
    if settings.data_sources and settings.data_sources.tables:
        table_lines = ["## Available Tables"]
        for table in settings.data_sources.tables:
            table_lines.append(f"\n### {table.identifier}")
            if table.column_configs:
                cols = ", ".join(c.column_name for c in table.column_configs)
                table_lines.append(f"Columns: {cols}")
        sections.append("\n".join(table_lines))

    # Format documentation with examples
    sections.append(
        """## YAML Format Documentation

### example_question_sqls
Pairs of natural-language questions with their correct SQL. Genie uses these as few-shot examples.
```yaml
example_question_sqls:
  - id: eq1
    question:
      - What is the total revenue by region?
    sql:
      - SELECT region, SUM(revenue) FROM sales GROUP BY region
```

### join_specs
Specify how tables should be joined. Each entry names a left table, right table, and the SQL join condition.
```yaml
join_specs:
  - id: js1
    left:
      identifier: catalog.schema.orders
      alias: o
    right:
      identifier: catalog.schema.customers
      alias: c
    sql:
      - o.customer_id = c.id
```

### sql_snippets
Reusable SQL fragments for filters, expressions, and measures. Include an alias and optional synonyms so Genie can match user intent.
```yaml
sql_snippets:
  filters:
    - id: f1
      alias: active_customers
      display_name: Active Customers
      synonyms:
        - current customers
      sql:
        - status = 'active'
  expressions:
    - id: e1
      alias: full_name
      display_name: Full Name
      sql:
        - CONCAT(first_name, ' ', last_name)
  measures:
    - id: m1
      alias: total_revenue
      display_name: Total Revenue
      synonyms:
        - revenue
        - sales
      sql:
        - SUM(revenue)
```

## Priority Guidance
- PREFER creating structured artifacts (example_question_sqls, join_specs, sql_snippets) over adding verbose text_instructions.
- text_instructions should be kept short and only cover guidance that CANNOT be expressed as structured artifacts.
- When tables need to be joined, always create a join_spec.
- When users commonly ask about a metric or filter, create a sql_snippet.
- When you know the correct SQL for a question, add it as an example_question_sql."""
    )

    return "\n\n".join(sections)


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

    # Always include empty scaffolding so the LLM sees these keys are available
    components.setdefault("example_question_sqls", [])
    components.setdefault("join_specs", [])
    components.setdefault(
        "sql_snippets", {"filters": [], "expressions": [], "measures": []}
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

    background = _build_background(settings)

    result = optimize_anything(
        seed_candidate=seed,
        evaluator=evaluator,
        objective=(
            "Optimise the Genie Space configuration so that Genie produces correct "
            "SQL answers to the benchmark questions. Focus on creating structured "
            "artifacts: create join_specs for table relationships, sql_snippets for "
            "reusable filters/expressions/measures, and example_question_sqls for "
            "common queries with their correct SQL. Keep text_instructions minimal — "
            "only for guidance that cannot be expressed as structured artifacts. "
            "Assume Databricks SQL dialect."
        ),
        background=background,
        config=GEPAConfig(
            engine=EngineConfig(max_metric_calls=max_evals),
            reflection=ReflectionConfig(reflection_lm="databricks/databricks-gpt-5-1"),
        ),
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
