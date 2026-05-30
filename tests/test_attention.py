"""Locked-in regression tests for the published numbers.

Every value asserted here is also printed in `demo.py`'s docstring and rendered
in `PIPELINE.md`. If a test breaks, the prose explanation in `blog.md` is now
inconsistent with the code — fix the code, not the test, unless you also
update the prose and PIPELINE.md.
"""

from __future__ import annotations

import math

import pytest

import app
import demo


# Tolerance for floating-point comparisons. The published numbers are rounded
# to 4 decimal places, so 1e-3 absolute is plenty without being brittle.
TOL = 1e-3


# ---------------------------------------------------------------------------
# Math primitives
# ---------------------------------------------------------------------------

class TestPrimitives:
    def test_dot_product(self) -> None:
        assert demo.dot_product([1, 2, 3], [4, 5, 6]) == 32
        assert app.dot([1, 2, 3], [4, 5, 6]) == 32

    def test_softmax_uniform(self) -> None:
        probs = app.softmax([0.0, 0.0, 0.0])
        assert probs == pytest.approx([1 / 3, 1 / 3, 1 / 3], abs=TOL)
        assert sum(probs) == pytest.approx(1.0, abs=1e-9)

    def test_softmax_sums_to_one(self) -> None:
        probs = app.softmax([2.7, 1.6, -0.3, 5.0])
        assert sum(probs) == pytest.approx(1.0, abs=1e-9)
        assert all(0.0 <= p <= 1.0 for p in probs)

    def test_softmax_temperature_flattens(self) -> None:
        sharp = app.softmax([2.0, 0.0], temperature=0.5)
        flat = app.softmax([2.0, 0.0], temperature=4.0)
        # Higher temperature -> distribution closer to uniform (0.5 / 0.5).
        assert abs(sharp[0] - 0.5) > abs(flat[0] - 0.5)

    def test_softmax_numerically_stable(self) -> None:
        probs = app.softmax([1000.0, 1000.0, 999.0])
        assert sum(probs) == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Inference — Scenario A: "the fastest database is"
# ---------------------------------------------------------------------------

class TestScenarioFastest:
    PROMPT = ["the", "fastest", "database", "is"]

    @pytest.fixture(scope="class")
    def result(self) -> dict:
        return app.run_inference(self.PROMPT)

    def test_predicts_redis(self, result: dict) -> None:
        assert result["predicted_token"] == "Redis"

    def test_probabilities(self, result: dict) -> None:
        assert result["candidates"] == ["Redis", "Postgres"]
        assert result["probabilities"][0] == pytest.approx(0.7425, abs=TOL)
        assert result["probabilities"][1] == pytest.approx(0.2575, abs=TOL)

    def test_attention_weights(self, result: dict) -> None:
        assert result["attention_weights"] == pytest.approx(
            [0.1825, 0.2447, 0.3929, 0.1799], abs=TOL
        )

    def test_context_vector(self, result: dict) -> None:
        assert result["context"] == pytest.approx(
            [0.2813, 0.0460, 0.3222, 0.3047], abs=TOL
        )

    def test_logits(self, result: dict) -> None:
        assert result["logits"] == pytest.approx([2.701, 1.642], abs=TOL)


# ---------------------------------------------------------------------------
# Inference — Scenario B: "the safest database is"
# ---------------------------------------------------------------------------

class TestScenarioSafest:
    PROMPT = ["the", "safest", "database", "is"]

    @pytest.fixture(scope="class")
    def result(self) -> dict:
        return app.run_inference(self.PROMPT)

    def test_predicts_postgres(self, result: dict) -> None:
        assert result["predicted_token"] == "Postgres"

    def test_probabilities(self, result: dict) -> None:
        assert result["probabilities"][0] == pytest.approx(0.3073, abs=TOL)
        assert result["probabilities"][1] == pytest.approx(0.6927, abs=TOL)

    def test_context_vector(self, result: dict) -> None:
        assert result["context"] == pytest.approx(
            [0.073, 0.254, 0.322, 0.305], abs=TOL
        )

    def test_logits(self, result: dict) -> None:
        assert result["logits"] == pytest.approx([1.765, 2.578], abs=TOL)


# ---------------------------------------------------------------------------
# The headline teaching point: same attention, different values -> different
# context -> different answer. The attention weights MUST match across the
# two scenarios, because they depend on K/Q dot products, and the K/Q
# projections of "fastest" and "safest" sit on different axes but get scored
# identically by the (symmetric) Query from the "is" position.
# ---------------------------------------------------------------------------

def test_attention_is_identical_across_scenarios() -> None:
    a = app.run_inference(["the", "fastest", "database", "is"])
    b = app.run_inference(["the", "safest", "database", "is"])
    assert a["attention_weights"] == pytest.approx(b["attention_weights"], abs=TOL)


# ---------------------------------------------------------------------------
# Cross-consistency: demo.py (the CLI) and app.py (the web API) must agree.
# ---------------------------------------------------------------------------

class TestDemoAppAgreement:
    @pytest.mark.parametrize(
        "prompt",
        [
            ["the", "fastest", "database", "is"],
            ["the", "safest", "database", "is"],
            ["the", "best", "database", "is"],
        ],
    )
    def test_inference_matches(self, prompt: list[str]) -> None:
        _, demo_probs, demo_attn = demo.run_inference(prompt, verbose=False)
        app_result = app.run_inference(prompt)
        assert list(demo_probs) == pytest.approx(
            app_result["probabilities"], abs=TOL
        )
        assert list(demo_attn) == pytest.approx(
            app_result["attention_weights"], abs=TOL
        )


# ---------------------------------------------------------------------------
# Training — "the best database is" -> Postgres, lr=5.0
# Numbers reproduced from demo.py's docstring.
# ---------------------------------------------------------------------------

class TestTraining:
    PROMPT = ["the", "best", "database", "is"]

    @pytest.fixture(scope="class")
    def result(self) -> dict:
        return app.train_output_weights(self.PROMPT, "Postgres", epochs=20)

    def test_epoch_1_starts_at_50_50(self, result: dict) -> None:
        step = result["history"][0]
        assert step["epoch"] == 1
        assert step["probabilities"] == pytest.approx([0.5, 0.5], abs=TOL)
        # -log(0.5) = 0.6931...
        assert step["loss"] == pytest.approx(math.log(2), abs=TOL)

    def test_epoch_5(self, result: dict) -> None:
        step = result["history"][4]
        assert step["epoch"] == 5
        assert step["probabilities"] == pytest.approx([0.0653, 0.9347], abs=TOL)
        assert step["loss"] == pytest.approx(0.0675, abs=TOL)

    def test_epoch_10(self, result: dict) -> None:
        step = result["history"][9]
        assert step["probabilities"] == pytest.approx([0.0323, 0.9677], abs=TOL)
        assert step["loss"] == pytest.approx(0.0328, abs=TOL)

    def test_epoch_20_converges(self, result: dict) -> None:
        step = result["history"][-1]
        assert step["epoch"] == 20
        assert step["probabilities"][1] >= 0.98
        assert step["loss"] <= 0.02

    def test_loss_is_monotonically_decreasing(self, result: dict) -> None:
        losses = [h["loss"] for h in result["history"]]
        for prev, curr in zip(losses, losses[1:]):
            assert curr <= prev + 1e-9, f"loss went up: {prev} -> {curr}"

    def test_learned_weights_are_antisymmetric(self, result: dict) -> None:
        # Starting from zero with a one-hot target, the gradient update for
        # Redis is exactly the negative of the gradient update for Postgres
        # at every step, so the learned weight rows must be mirror images.
        w = result["learned_weights"]
        for r, p in zip(w["Redis"], w["Postgres"]):
            assert r == pytest.approx(-p, abs=TOL)

    def test_learned_weights_match_demo(self, result: dict) -> None:
        w = result["learned_weights"]
        assert w["Redis"] == pytest.approx(
            [-1.348, -1.170, -2.448, -2.236], abs=TOL
        )
        assert w["Postgres"] == pytest.approx(
            [1.348, 1.170, 2.448, 2.236], abs=TOL
        )


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrors:
    def test_unknown_token_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown tokens"):
            app.run_inference(["the", "quickest", "database", "is"])

    def test_unknown_target_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown target"):
            app.train_output_weights(
                ["the", "best", "database", "is"], "MongoDB"
            )


# ---------------------------------------------------------------------------
# Dataset / repeated-example training
#
# These tests lock in the "small but powerful" story: the dataset is just a
# list of sentences, repeating the same one many times is enough to push the
# model's output weights toward the repeated answer, and no other path
# (attention, embeddings, candidate set) needs to change.
# ---------------------------------------------------------------------------


class TestParseExample:
    def test_basic_split(self) -> None:
        prompt, target = app.parse_example("the fastest database is Redis")
        assert prompt == ["the", "fastest", "database", "is"]
        assert target == "Redis"

    def test_strips_whitespace(self) -> None:
        prompt, target = app.parse_example("  the  best   database  is  Postgres  ")
        assert prompt == ["the", "best", "database", "is"]
        assert target == "Postgres"

    def test_short_example_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least 2 tokens"):
            app.parse_example("Redis")


class TestLoadDataset:
    def test_loads_repeated_fastest(self) -> None:
        data = app.load_dataset("repeated_fastest")
        assert data["name"] == "repeated_fastest"
        assert len(data["examples"]) == 20
        assert all(ex["text"] == "the fastest database is Redis" for ex in data["examples"])

    def test_loads_repeated_safest(self) -> None:
        data = app.load_dataset("repeated_safest")
        assert len(data["examples"]) == 20
        assert all(ex["text"] == "the safest database is Postgres" for ex in data["examples"])

    def test_missing_dataset_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            app.load_dataset("does_not_exist")

    def test_invalid_name_rejected(self) -> None:
        with pytest.raises(ValueError, match="Invalid dataset name"):
            app.load_dataset("../etc/passwd")

    def test_list_datasets_includes_known(self) -> None:
        names = {ds["name"] for ds in app.list_datasets()}
        assert "repeated_fastest" in names
        assert "repeated_safest" in names


class TestRepeatedExampleTraining:
    """The headline teaching point.

    Repeating the same example many times pushes P(target) from 50/50 toward
    the target. No attention re-routing is needed — same context vector every
    step, same error signal, weights accumulate the same nudge.
    """

    def test_first_step_is_uniform(self) -> None:
        result = app.train_output_weights_from_examples(
            [{"text": "the fastest database is Redis"}]
        )
        first = result["history"][0]
        assert first["probabilities"] == pytest.approx([0.5, 0.5], abs=TOL)
        # Loss values are stored rounded to 6 dp for the UI; that's far tighter
        # than the published numbers, so 1e-5 absolute is the right scale.
        assert first["loss"] == pytest.approx(math.log(2), abs=1e-5)

    def test_repetition_pushes_probability_toward_target(self) -> None:
        examples = [{"text": "the fastest database is Redis"}] * 20
        result = app.train_output_weights_from_examples(examples)

        first_p_redis = result["history"][0]["probabilities"][0]
        last_p_redis = result["history"][-1]["probabilities"][0]
        final_p_redis = result["final"]["probabilities"][0]

        assert first_p_redis == pytest.approx(0.5, abs=TOL)
        # After 20 repetitions the target should dominate.
        assert last_p_redis > 0.95
        assert final_p_redis > 0.95

    def test_loss_is_monotonically_decreasing(self) -> None:
        examples = [{"text": "the fastest database is Redis"}] * 20
        result = app.train_output_weights_from_examples(examples)
        losses = [h["loss"] for h in result["history"]]
        for prev, curr in zip(losses, losses[1:]):
            assert curr <= prev + 1e-9, f"loss went up: {prev} -> {curr}"

    def test_target_counts_track_repetition(self) -> None:
        examples = [{"text": "the fastest database is Redis"}] * 7
        result = app.train_output_weights_from_examples(examples)
        assert result["example_count"] == 7
        assert result["target_counts"]["Redis"] == 7
        assert result["target_counts"]["Postgres"] == 0
        assert result["unique_prompts"] == ["the fastest database is"]

    def test_repeated_safest_pushes_postgres(self) -> None:
        examples = [{"text": "the safest database is Postgres"}] * 20
        result = app.train_output_weights_from_examples(examples)
        p_redis, p_postgres = result["final"]["probabilities"]
        assert p_postgres > 0.95
        assert p_redis < 0.05

    def test_empty_dataset_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            app.train_output_weights_from_examples([])

    def test_unknown_target_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown target"):
            app.train_output_weights_from_examples(
                [{"text": "the fastest database is MongoDB"}]
            )

    def test_unknown_prompt_token_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown tokens"):
            app.train_output_weights_from_examples(
                [{"text": "the quickest database is Redis"}]
            )

    def test_dataset_file_drives_the_story(self) -> None:
        """End-to-end check: the JSON file on disk produces the headline result."""
        data = app.load_dataset("repeated_fastest")
        result = app.train_output_weights_from_examples(data["examples"])
        assert result["example_count"] == 20
        assert result["final"]["target"] == "Redis"
        assert result["final"]["probabilities"][0] > 0.95


# ---------------------------------------------------------------------------
# Markdown renderer — minimal sanity checks (HTML produced by app.py is what
# the FastAPI route ships to the browser).
# ---------------------------------------------------------------------------

class TestMarkdownRenderer:
    def test_heading_gets_id_slug(self) -> None:
        html = app.markdown_to_html("### The Brain's Memory: Model Weights")
        assert html == '<h3 id="the-brain-s-memory-model-weights">The Brain\'s Memory: Model Weights</h3>'

    def test_blockquote(self) -> None:
        html = app.markdown_to_html("> Hello **world**")
        assert "<blockquote>" in html
        assert "<strong>world</strong>" in html

    def test_link_gets_target_blank(self) -> None:
        html = app.markdown_to_html("See [the repo](https://github.com/x/y).")
        assert 'href="https://github.com/x/y"' in html
        assert 'target="_blank"' in html
        assert 'rel="noopener noreferrer"' in html

    def test_inline_code(self) -> None:
        html = app.markdown_to_html("Run `python3 demo.py` to start.")
        assert "<code>python3 demo.py</code>" in html

    def test_bold_and_italic(self) -> None:
        html = app.markdown_to_html("**bold** and *italic*")
        assert "<strong>bold</strong>" in html
        assert "<em>italic</em>" in html

    def test_horizontal_rule(self) -> None:
        html = app.markdown_to_html("paragraph\n\n---\n\nnext")
        assert "<hr />" in html

    def test_renders_real_blog(self) -> None:
        html = app.load_blog_html()
        assert "<h1" in html
        assert "Demystifying" in html
        # GitHub-canonical links (set up in the previous commit).
        assert "github.com/ranfysvalle02/attention-and-weights" in html
        # No leftover local refs.
        assert 'href="demo.py"' not in html
        assert 'href="app.py"' not in html
