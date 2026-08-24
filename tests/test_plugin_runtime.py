from __future__ import annotations

from pathlib import Path

import pytest

from core.services.plugins import ManimVideoPlugin, PluginJobRequest, PluginRuntime
from core.services.plugins.runtime import PluginJobResult


class _DummyInference:
    max_tokens = 1200

    @staticmethod
    def is_configured() -> bool:
        return False

    @staticmethod
    def chat_completions(messages, max_tokens=None):
        raise RuntimeError("should not be called when inference is not configured")

    @staticmethod
    def extract_response_payload(response):
        return "", ""


class _GenericPlanInference:
    max_tokens = 1200

    @staticmethod
    def is_configured() -> bool:
        return True

    @staticmethod
    def chat_completions(messages, max_tokens=None):
        class _Msg:
            content = (
                '{"title":"Step-by-step concept walkthrough",'
                '"learning_goal":"Solve the question",'
                '"formula_latex":"Write the key formula first, then substitute values step by step.",'
                '"steps":["Identify the known values and what must be found.",'
                '"Write the core formula clearly before calculation.",'
                '"Substitute values carefully and simplify line by line.",'
                '"Check units and verify the final answer is reasonable."],'
                '"worked_example":["Given values from the question.",'
                '"Use formula: Write the key formula first.",'
                '"Compute each step and present the final result clearly."],'
                '"visual_focus":"generic",'
                '"answer_line":"The answer follows from applying the formula step by step."}'
            )
            reasoning_content = None

        class _Choice:
            message = _Msg()

        class _Resp:
            choices = [_Choice()]

        return _Resp()

    @staticmethod
    def extract_response_payload(response):
        return response.choices[0].message.content, ""


class _SimplePlugin:
    plugin_id = "simple"

    async def run(self, request: PluginJobRequest, emit):
        await emit("planning", "ok")
        return PluginJobResult(plan_text="done")


@pytest.mark.asyncio
async def test_plugin_runtime_routes_to_registered_handler(tmp_path: Path):
    runtime = PluginRuntime(artifact_root=tmp_path / "artifacts")
    runtime.register(_SimplePlugin())
    events: list[tuple[str, str]] = []

    async def emit(phase: str, message: str):
        events.append((phase, message))

    req = PluginJobRequest(
        job_id="job-1",
        plugin_id="simple",
        query="q",
        context_text="ctx",
        mode="environment",
        current_page=1,
        book_id=None,
        output_dir=runtime.create_job_dir("job-1"),
    )
    result = await runtime.run_job(req, emit)
    assert result.plan_text == "done"
    assert events == [("planning", "ok")]


@pytest.mark.asyncio
async def test_plugin_runtime_rejects_unknown_plugin(tmp_path: Path):
    runtime = PluginRuntime(artifact_root=tmp_path / "artifacts")
    req = PluginJobRequest(
        job_id="job-unknown",
        plugin_id="missing",
        query="q",
        context_text="ctx",
        mode="environment",
        current_page=1,
        book_id=None,
        output_dir=runtime.create_job_dir("job-unknown"),
    )

    async def emit(_phase: str, _message: str):
        return

    with pytest.raises(ValueError):
        await runtime.run_job(req, emit)


@pytest.mark.asyncio
async def test_manim_video_plugin_fallback_generates_script_and_video(tmp_path: Path):
    plugin = ManimVideoPlugin(_DummyInference(), skill_root=tmp_path / "missing-skill")
    output_dir = tmp_path / "job"
    output_dir.mkdir(parents=True, exist_ok=True)

    fake_video = output_dir / "media" / "lesson.mp4"
    fake_video.parent.mkdir(parents=True, exist_ok=True)
    fake_video.write_bytes(b"video")

    def _fake_render(script_path: Path, media_dir: Path) -> Path:
        assert script_path.exists()
        assert media_dir.exists()
        return fake_video

    plugin._render = _fake_render  # type: ignore[attr-defined]

    events = []

    async def emit(phase: str, message: str):
        events.append((phase, message))

    request = PluginJobRequest(
        job_id="job-2",
        plugin_id="manim_video",
        query="Explain quadratic roots",
        context_text="Quadratic equations and graph roots.",
        mode="environment",
        current_page=4,
        book_id=None,
        output_dir=output_dir,
    )
    result = await plugin.run(request, emit)

    assert result.script_path is not None
    assert result.video_path is not None
    script_text = Path(result.script_path).read_text(encoding="utf-8")
    assert "class LessonScene(Scene)" in script_text
    assert Path(result.video_path).exists()
    assert events[0][0] == "planning"


def test_manim_plugin_resolves_cli_from_path(monkeypatch, tmp_path: Path):
    fake_manim = tmp_path / "manim"
    fake_manim.write_text("#!/bin/sh\n", encoding="utf-8")
    fake_manim.chmod(0o755)

    monkeypatch.setattr("core.services.plugins.manim_video_plugin.shutil.which", lambda _name: str(fake_manim))

    resolved = ManimVideoPlugin._resolve_manim_cli()
    assert resolved == str(fake_manim)


def test_extract_python_block_handles_unclosed_fence():
    text = "```python\nfrom manim import *\nclass LessonScene(Scene):\n    pass\n"
    extracted = ManimVideoPlugin._extract_python_block(text)
    assert extracted.startswith("from manim import *")
    assert "```" not in extracted


def test_script_validation_rejects_invalid_python():
    invalid = "from manim import *\nclass LessonScene(Scene):\n    def construct(self):\n        x ="
    assert not ManimVideoPlugin._script_looks_valid(invalid, "LessonScene")


def test_fallback_script_is_valid_with_multiline_query():
    plugin = ManimVideoPlugin(_DummyInference())
    query = "could you show step-by-step\nformula for volume of sphere"
    plan = plugin._fallback_plan(query, "formula and geometry context")
    script = plugin._template_script_from_plan(query, plan)
    assert "class LessonScene(Scene):" in script
    assert plugin._script_looks_valid(script, "LessonScene")


def test_plan_generation_returns_steps():
    plugin = ManimVideoPlugin(_DummyInference())
    plan, mode = plugin._generate_plan(
        "show area of scalene triangle with steps",
        "Area of triangle uses half times base times height.",
    )
    assert mode == "inference_unavailable_fallback"
    assert isinstance(plan.get("steps"), list)
    assert len(plan["steps"]) >= 4


def test_rate_fallback_pattern_handles_throughput():
    plugin = ManimVideoPlugin(_DummyInference())
    plan = plugin._fallback_plan(
        "explain throughput",
        "Throughput means the actual amount of data sent or received over a network.",
    )
    script = plugin._template_script_from_plan("explain throughput", plan)

    assert plan["title"] == "Rate and Capacity"
    assert plan["visual_focus"] == "network"
    assert "successfully transferred" in plan["formula_latex"]
    assert "successful data / second" in script
    assert "Write the key formula first" not in script
    assert plugin._script_looks_valid(script, "LessonScene")


def test_generic_llm_plan_falls_back_to_rate_pattern():
    plugin = ManimVideoPlugin(_GenericPlanInference())

    plan, mode = plugin._generate_plan(
        "explain throughput",
        "Throughput is actual data successfully transferred per second.",
    )

    assert mode == "plan_quality_fallback"
    assert plan["title"] == "Rate and Capacity"
    assert "Throughput =" in plan["formula_latex"]


def test_identity_fallback_uses_relationship_pattern():
    plugin = ManimVideoPlugin(_DummyInference())

    plan = plugin._fallback_plan(
        "Explain conditional identities in Nepali:",
        "Conditional Identities: tan A = cot B is true when A + B = 90°.",
    )
    script = plugin._template_script_from_plan("Explain conditional identities in Nepali:", plan)

    assert plan["title"] == "Conditional Relationship"
    assert plan["visual_focus"] == "identity"
    assert "A + B = 90" in plan["formula_latex"]
    assert "Write the key formula first" not in script
    assert "Given values from the question" not in script
    assert "condition: A + B = 90" in script
    assert plugin._script_looks_valid(script, "LessonScene")


def test_optimization_fallback_pattern_handles_linear_programming():
    plugin = ManimVideoPlugin(_DummyInference())

    plan = plugin._fallback_plan(
        "Explain Linear programming in Nepali",
        "रेखीय योजना (Linear Programming) uses linear inequalities and an objective function.",
    )
    script = plugin._template_script_from_plan("Explain Linear programming in Nepali", plan)

    assert plan["title"] == "Optimization With Constraints"
    assert plan["visual_focus"] == "optimization"
    assert "Max/Min Z" in plan["formula_latex"]
    assert "feasible region" in script
    assert "maximize Z" in script
    assert "best corner" in script
    assert "Write the key formula first" not in script
    assert "Given values from the question" not in script
    assert plugin._script_looks_valid(script, "LessonScene")


def test_optimization_fallback_pattern_handles_maximize_prompt():
    plugin = ManimVideoPlugin(_DummyInference())

    plan = plugin._fallback_plan(
        "How do I maximize profit with constraints?",
        "A business chooses two products with limited labor and material.",
    )

    assert plan["title"] == "Optimization With Constraints"
    assert plan["visual_focus"] == "optimization"


def test_generic_llm_plan_falls_back_to_conditional_identity_plan():
    plugin = ManimVideoPlugin(_GenericPlanInference())

    plan, mode = plugin._generate_plan(
        "Explain conditional identities in Nepali:",
        "tan A = cot B is true when A + B = 90°.",
    )

    assert mode == "plan_quality_fallback"
    assert plan["title"] == "Conditional Relationship"
    assert plan["visual_focus"] == "identity"


def test_generic_llm_plan_falls_back_to_optimization_pattern():
    plugin = ManimVideoPlugin(_GenericPlanInference())

    plan, mode = plugin._generate_plan(
        "Explain Linear programming in Nepali",
        "रेखीय योजना (Linear Programming) solves optimization with linear inequalities.",
    )

    assert mode == "plan_quality_fallback"
    assert plan["title"] == "Optimization With Constraints"
    assert plan["visual_focus"] == "optimization"
