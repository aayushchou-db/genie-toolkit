import json
from unittest.mock import MagicMock, patch

import pytest

from genie_toolkit.optimiser import (
    _build_background,
    build_evaluator,
    deserialise_config_components,
    llm_judge,
    serialise_config_components,
    split_benchmarks,
)
from genie_toolkit.schemas import (
    GenieBenchmarkAnswer,
    GenieBenchmarkQuestion,
    GenieColumnConfig,
    GenieDataSources,
    GenieExampleSQL,
    GenieInstructions,
    GenieSchemaSettings,
    GenieTableConfig,
    GenieTextInstruction,
)


def _make_questions(n: int) -> list[GenieBenchmarkQuestion]:
    return [
        GenieBenchmarkQuestion(
            id=str(i),
            question=[f"Question {i}?"],
            answer=[GenieBenchmarkAnswer(format="text", content=[f"Answer {i}"])],
        )
        for i in range(n)
    ]


class TestSplitBenchmarks:
    def test_correct_ratios(self):
        questions = _make_questions(10)
        train, val, test = split_benchmarks(questions, 0.7, 0.15)
        assert len(train) == 7
        assert len(val) == 1  # int(10 * 0.85) - 7 = 1
        assert len(test) == 2

    def test_reproducibility(self):
        questions = _make_questions(10)
        train1, val1, test1 = split_benchmarks(questions, 0.7, 0.15)
        train2, val2, test2 = split_benchmarks(questions, 0.7, 0.15)
        assert [q.id for q in train1] == [q.id for q in train2]
        assert [q.id for q in val1] == [q.id for q in val2]
        assert [q.id for q in test1] == [q.id for q in test2]

    def test_no_overlap(self):
        questions = _make_questions(20)
        train, val, test = split_benchmarks(questions, 0.7, 0.15)
        all_ids = [q.id for q in train + val + test]
        assert len(all_ids) == len(set(all_ids))
        assert len(all_ids) == 20

    def test_too_few_questions_raises(self):
        questions = _make_questions(2)
        with pytest.raises(ValueError, match="at least 3"):
            split_benchmarks(questions, 0.7, 0.15)

    def test_invalid_ratios_raises(self):
        questions = _make_questions(10)
        with pytest.raises(ValueError, match="<= 1.0"):
            split_benchmarks(questions, 0.8, 0.3)

    def test_minimum_three_questions(self):
        questions = _make_questions(3)
        train, val, test = split_benchmarks(questions, 0.7, 0.15)
        assert len(train) >= 1
        assert len(val) >= 1
        # All questions accounted for
        assert len(train) + len(val) + len(test) == 3


class TestSerialiseDeserialise:
    def test_round_trip_with_text_instructions(self):
        instructions = GenieInstructions(
            text_instructions=[
                GenieTextInstruction(id="ti1", content=["Be helpful", "Be concise"])
            ]
        )
        serialised = serialise_config_components(instructions)
        result_instructions, result_config = deserialise_config_components(serialised)

        assert result_instructions.text_instructions is not None
        assert len(result_instructions.text_instructions) == 1
        assert result_instructions.text_instructions[0].content == [
            "Be helpful",
            "Be concise",
        ]
        assert result_config is None

    def test_round_trip_with_example_sqls(self):
        instructions = GenieInstructions(
            example_question_sqls=[
                GenieExampleSQL(
                    id="eq1",
                    question=["How many users?"],
                    sql=["SELECT COUNT(*) FROM users"],
                )
            ]
        )
        serialised = serialise_config_components(instructions)
        result_instructions, _ = deserialise_config_components(serialised)

        assert result_instructions.example_question_sqls is not None
        assert len(result_instructions.example_question_sqls) == 1

    def test_round_trip_with_sample_questions(self):
        instructions = GenieInstructions()
        config = {"sample_questions": [{"question": ["What is revenue?"]}]}
        serialised = serialise_config_components(instructions, config)
        _, result_config = deserialise_config_components(serialised)

        assert result_config is not None
        assert result_config["sample_questions"][0]["question"] == ["What is revenue?"]

    def test_empty_instructions_includes_scaffolding(self):
        instructions = GenieInstructions()
        serialised = serialise_config_components(instructions)
        result_instructions, result_config = deserialise_config_components(serialised)
        assert result_instructions.text_instructions is None
        assert result_config is None

        # Scaffolding keys must be present in the serialised YAML
        assert "example_question_sqls:" in serialised
        assert "join_specs:" in serialised
        assert "sql_snippets:" in serialised
        assert "filters:" in serialised
        assert "expressions:" in serialised
        assert "measures:" in serialised

    def test_scaffolding_does_not_overwrite_existing(self):
        instructions = GenieInstructions(
            example_question_sqls=[
                GenieExampleSQL(
                    id="eq1",
                    question=["How many users?"],
                    sql=["SELECT COUNT(*) FROM users"],
                )
            ]
        )
        serialised = serialise_config_components(instructions)
        result_instructions, _ = deserialise_config_components(serialised)

        assert len(result_instructions.example_question_sqls) == 1
        assert result_instructions.example_question_sqls[0].id == "eq1"

    def test_deserialise_invalid_yaml(self):
        # Invalid YAML should not crash, returns empty instructions
        result_instructions, result_config = deserialise_config_components("")
        assert result_instructions.text_instructions is None
        assert result_config is None


class TestLLMJudge:
    def test_score_1_on_equivalent(self):
        mock_wc = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(
            {"score": 1, "reason": "Semantically equivalent"}
        )
        mock_wc.serving_endpoints.query.return_value = mock_response

        score, reason = llm_judge(
            mock_wc, "How many users?", "42", "42 users", "test-endpoint"
        )
        assert score == 1
        assert "equivalent" in reason.lower()

    def test_score_0_on_not_equivalent(self):
        mock_wc = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(
            {"score": 0, "reason": "Completely different"}
        )
        mock_wc.serving_endpoints.query.return_value = mock_response

        score, reason = llm_judge(
            mock_wc, "How many users?", "42", "unknown", "test-endpoint"
        )
        assert score == 0

    def test_handles_llm_error(self):
        mock_wc = MagicMock()
        mock_wc.serving_endpoints.query.side_effect = Exception("API Error")

        score, reason = llm_judge(
            mock_wc, "Q?", "expected", "actual", "test-endpoint"
        )
        assert score == 0
        assert "error" in reason.lower()

    def test_clamps_score(self):
        mock_wc = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(
            {"score": 5, "reason": "oops"}
        )
        mock_wc.serving_endpoints.query.return_value = mock_response

        score, _ = llm_judge(mock_wc, "Q?", "A", "A", "ep")
        assert score == 1


class TestBuildEvaluator:
    @patch("genie_toolkit.optimiser.query_genie_space")
    @patch("genie_toolkit.optimiser.llm_judge")
    def test_evaluator_scores_correctly(self, mock_judge, mock_query):
        mock_judge.return_value = (1, "correct")
        mock_query.return_value = "Answer 0"

        mock_service = MagicMock()
        mock_wc = MagicMock()

        settings = GenieSchemaSettings(
            version=1,
            instructions=GenieInstructions(
                text_instructions=[
                    GenieTextInstruction(id="ti1", content=["Be helpful"])
                ]
            ),
        )

        questions = _make_questions(2)
        evaluator = build_evaluator(
            mock_service, "sp123", questions, mock_wc, "test-ep", settings
        )

        candidate = serialise_config_components(settings.instructions)
        score = evaluator(candidate)

        assert score == 1.0
        assert mock_service.update.called

    @patch("genie_toolkit.optimiser.query_genie_space")
    @patch("genie_toolkit.optimiser.llm_judge")
    def test_evaluator_handles_parse_error(self, mock_judge, mock_query):
        mock_service = MagicMock()
        settings = GenieSchemaSettings(version=1)

        evaluator = build_evaluator(
            mock_service, "sp123", [], MagicMock(), "ep", settings
        )

        # Invalid YAML that can't be parsed
        score = evaluator("{{{{invalid yaml: [")
        assert score == 0.0


class TestBuildBackground:
    def test_includes_table_identifiers(self):
        settings = GenieSchemaSettings(
            version=1,
            data_sources=GenieDataSources(
                tables=[
                    GenieTableConfig(identifier="catalog.schema.orders"),
                    GenieTableConfig(identifier="catalog.schema.customers"),
                ]
            ),
        )
        bg = _build_background(settings)
        assert "catalog.schema.orders" in bg
        assert "catalog.schema.customers" in bg

    def test_includes_column_names(self):
        settings = GenieSchemaSettings(
            version=1,
            data_sources=GenieDataSources(
                tables=[
                    GenieTableConfig(
                        identifier="catalog.schema.orders",
                        column_configs=[
                            GenieColumnConfig(column_name="order_id"),
                            GenieColumnConfig(column_name="total"),
                        ],
                    ),
                ]
            ),
        )
        bg = _build_background(settings)
        assert "order_id" in bg
        assert "total" in bg

    def test_includes_format_examples(self):
        settings = GenieSchemaSettings(version=1)
        bg = _build_background(settings)
        assert "example_question_sqls" in bg
        assert "join_specs" in bg
        assert "sql_snippets" in bg
        assert "Priority Guidance" in bg

    def test_no_tables_section_without_data_sources(self):
        settings = GenieSchemaSettings(version=1)
        bg = _build_background(settings)
        assert "Available Tables" not in bg


class TestRunOptimisation:
    @patch("genie_toolkit.optimiser.optimize_anything")
    @patch("genie_toolkit.optimiser._evaluate_questions")
    def test_passes_background_to_optimizer(self, mock_eval, mock_optimize):
        mock_result = MagicMock()
        mock_result.best_candidate = "example_question_sqls: []\njoin_specs: []\nsql_snippets:\n  filters: []\n  expressions: []\n  measures: []\n"
        mock_result.best_score = 0.8
        mock_optimize.return_value = mock_result
        mock_eval.return_value = 0.75

        mock_service = MagicMock()
        settings = GenieSchemaSettings(
            version=1,
            data_sources=GenieDataSources(
                tables=[GenieTableConfig(identifier="catalog.schema.orders")]
            ),
            instructions=GenieInstructions(),
        )
        questions = _make_questions(4)

        from genie_toolkit.optimiser import run_optimisation

        run_optimisation(
            mock_service, "sp123", settings, questions, "test-ep", max_evals=5
        )

        # Verify optimize_anything was called with background kwarg
        _, kwargs = mock_optimize.call_args
        assert "background" in kwargs
        assert "catalog.schema.orders" in kwargs["background"]
        assert "structured artifacts" in kwargs["objective"].lower()
