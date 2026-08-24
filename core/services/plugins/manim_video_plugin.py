from __future__ import annotations

import asyncio
import ast
import json
import re
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
from textwrap import dedent

from .runtime import EmitFn, PluginJobRequest, PluginJobResult


class ManimVideoPlugin:
    plugin_id = "manim_video"

    def __init__(
        self,
        inference_service,
        skill_root: str | Path = "manim-video",
        quality: str = "l",
        render_timeout_seconds: int = 180,
    ):
        self._inference = inference_service
        self._skill_root = Path(skill_root)
        self._quality = quality
        self._render_timeout_seconds = render_timeout_seconds
        self._scene_name = "LessonScene"

    @staticmethod
    def _clip(text: str, limit: int = 220) -> str:
        text = re.sub(r"\s+", " ", str(text or "").strip())
        if len(text) <= limit:
            return text
        return text[: max(0, limit - 3)].rstrip() + "..."

    def _load_skill_context(self) -> str:
        skill_doc = self._skill_root / "SKILL.md"
        scene_doc = self._skill_root / "references" / "scene-planning.md"

        sections: list[str] = []
        if skill_doc.exists():
            sections.append(skill_doc.read_text(encoding="utf-8")[:5000])
        if scene_doc.exists():
            sections.append(scene_doc.read_text(encoding="utf-8")[:2500])
        return "\n\n".join(sections).strip()

    def _plan_prompt(self, query: str, context_text: str, style_context: str) -> str:
        return dedent(
            f"""
            You are a lesson planner for a Nepali high-school tutoring app.
            Use this style guide context:
            {style_context}

            Student question:
            {query}

            Textbook context:
            {context_text}

            Build a concrete teaching blueprint focused on solving the question, not meta commentary.
            Return ONLY a JSON object (no markdown) with keys:
            - "title": short lesson title
            - "learning_goal": one sentence
            - "formula_latex": core formula in latex-like text (or plain formula if unsure)
            - "steps": array of 4 concise actionable steps
            - "worked_example": array of 3 to 5 short solution lines
            - "visual_focus": one of "triangle", "circle", "algebra", "generic"
            - "answer_line": one sentence with the direct answer idea
            """
        ).strip()

    def _script_prompt_from_plan(
        self,
        query: str,
        context_text: str,
        style_context: str,
        plan: dict[str, object],
    ) -> str:
        plan_json = json.dumps(plan, ensure_ascii=False, indent=2)
        return dedent(
            f"""
            You are generating a Manim Community Edition script for a high-school tutoring app.
            Use this style guide context:
            {style_context}

            Student query:
            {query}

            Structured teaching blueprint:
            {plan_json}

            Source context:
            {context_text}

            Requirements:
            - Return only valid Python code in one ```python fenced block.
            - Code must include `from manim import *`.
            - Define class `{self._scene_name}(Scene)`.
            - The animation must teach the actual solution flow (formula + worked example), not just planning text.
            - Keep script robust for low-quality render (`-ql`) and avoid fragile APIs.
            - Use readable text sizes (title >= 42, body >= 30).
            - Include at least 5 explicit `self.wait(...)` pauses.
            - Keep total duration around 18 to 45 seconds.
            - Prefer Text/MathTex, FadeIn/Write/Transform/Create only.
            """
        ).strip()

    @staticmethod
    def _extract_python_block(text: str) -> str:
        text = (text or "").strip()
        match = re.search(r"```python\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
        if match:
            text = match.group(1).strip()

        lines = text.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(line for line in lines if line.strip() not in {"```", "```python"}).strip()
        return cleaned or text.strip()

    @staticmethod
    def _extract_json_object(text: str) -> dict | None:
        text = (text or "").strip()
        if not text:
            return None

        fenced = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
        if fenced:
            text = fenced.group(1)
        else:
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                text = text[start : end + 1]

        try:
            payload = json.loads(text)
        except Exception:
            return None
        return payload if isinstance(payload, dict) else None

    @staticmethod
    def _wrap_text(text: str, width: int = 34, max_lines: int = 3) -> str:
        text = re.sub(r"\s+", " ", text.strip())
        lines = textwrap.wrap(text, width=width)
        if not lines:
            return ""
        if len(lines) > max_lines:
            lines = lines[:max_lines]
            if not lines[-1].endswith("..."):
                lines[-1] = lines[-1].rstrip(".") + "..."
        return "\n".join(lines)

    @staticmethod
    def _latex_to_text(expr: str) -> str:
        expr = (expr or "").strip().strip("$")
        expr = expr.replace("\\times", "×")
        expr = expr.replace("\\cdot", "·")
        expr = expr.replace("\\pi", "π")
        expr = expr.replace("\\sqrt", "sqrt")
        expr = re.sub(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"(\1)/(\2)", expr)
        expr = expr.replace("{", "").replace("}", "")
        expr = expr.replace("\\", "")
        expr = re.sub(r"\s+", " ", expr).strip()
        return expr or "Use the core formula from the lesson."

    @staticmethod
    def _matches_any(scope: str, terms: tuple[str, ...]) -> bool:
        return any(term in scope for term in terms)

    def _teaching_pattern(self, query: str, context_text: str) -> str:
        scope = f"{query} {context_text}".lower()
        if self._matches_any(
            scope,
            (
                "linear programming",
                "linear programme",
                "optimization",
                "optimisation",
                "maximize",
                "maximise",
                "minimize",
                "minimise",
                "objective function",
                "constraint",
                "linear inequalities",
                "feasible region",
                "रेखीय योजना",
            ),
        ):
            return "optimization"
        if self._matches_any(scope, ("throughput", "bandwidth", "rate", "per second", "capacity")):
            return "rate"
        if self._matches_any(scope, ("conditional identit", "condition identity", "identity", "lhs", "r.h.s", "l.h.s")):
            return "identity"
        if self._matches_any(scope, ("remainder theorem", "factor", "factorise", "factorize", "root", "zero", "polynomial")):
            return "algebra_roots"
        if self._matches_any(scope, ("triangle", "circle", "sphere", "area", "volume", "pythag")):
            return "geometry_measure"
        return "concept"

    def _fallback_formula(self, query: str, context_text: str) -> str:
        pattern = self._teaching_pattern(query, context_text)
        scope = f"{query} {context_text}".lower()
        if pattern == "rate" and "throughput" in scope:
            return "Throughput = data successfully transferred / time"
        if pattern == "rate" and "bandwidth" in scope:
            return "Bandwidth = maximum data capacity of a network link"
        if pattern == "rate":
            return "Rate = amount changed / time"
        if pattern == "identity":
            return "A + B = 90° भए tan A = cot B"
        if pattern == "optimization":
            return "Max/Min Z = ax + by, subject to linear constraints"
        if pattern == "algebra_roots" and "remainder theorem" in scope:
            return "p(x) लाई (x - a) ले भाग गर्दा remainder = p(a)"
        if pattern == "algebra_roots":
            return "Polynomial: ax^n + ... + c, जहाँ n whole number हो"
        if "scalene" in scope and "area" in scope:
            return r"A = \frac{1}{2} b h"
        if "area" in scope and "triangle" in scope:
            return r"A = \frac{1}{2} b h"
        if "volume" in scope and "sphere" in scope:
            return r"V = \frac{4}{3}\pi r^3"
        if "pythag" in scope or "right triangle" in scope:
            return r"c^2 = a^2 + b^2"
        if "simple interest" in scope or "interest" in scope:
            return r"I = P r t"
        return "Define the key relationship, then apply it step by step."

    def _pattern_fallback_plan(self, query: str, context_text: str) -> dict[str, object] | None:
        pattern = self._teaching_pattern(query, context_text)
        formula = self._fallback_formula(query, context_text)

        if pattern == "rate":
            return {
                "title": "Rate and Capacity",
                "learning_goal": "Explain how an amount changes or moves over time, and compare expected capacity with actual rate.",
                "formula_latex": formula,
                "steps": [
                    "Identify the quantity being measured and the time interval.",
                    "Separate maximum capacity from the amount that actually succeeds.",
                    "Divide successful amount by time to get the real rate.",
                    "Use units like per second to compare different situations.",
                ],
                "worked_example": [
                    "If 20 units arrive successfully in 2 seconds, measure the real delivery rate.",
                    "Rate = 20 units / 2 seconds.",
                    "Rate = 10 units per second, so the actual delivery is 10 each second.",
                ],
                "visual_focus": "network",
                "answer_line": "Rate compares a successful amount with the time it takes.",
            }

        if pattern == "identity":
            return {
                "title": "Conditional Relationship",
                "learning_goal": "Understand when two expressions become equal under a stated condition.",
                "formula_latex": formula,
                "steps": [
                    "Write the condition that must be true first.",
                    "Place the left-hand side and right-hand side side by side.",
                    "Substitute a simple value pair that satisfies the condition.",
                    "Check whether both sides match under that condition.",
                ],
                "worked_example": [
                    "Choose values that satisfy the condition.",
                    "Evaluate the left side and the right side separately.",
                    "If both sides match, the relationship holds for that condition.",
                ],
                "visual_focus": "identity",
                "answer_line": "A conditional relationship is true only when its condition is satisfied.",
            }

        if pattern == "optimization":
            return {
                "title": "Optimization With Constraints",
                "learning_goal": "Model choices with constraints and choose the best feasible option.",
                "formula_latex": formula,
                "steps": [
                    "Choose decision variables such as x and y for the two quantities.",
                    "Write the target expression that should be maximized or minimized.",
                    "Translate every limit or condition into a constraint.",
                    "Represent the feasible choices and identify boundary points.",
                    "Evaluate the target expression at candidate points and select the best one.",
                ],
                "worked_example": [
                    "Maximize Z = 3x + 2y with x + y <= 6, x <= 4, y <= 3.",
                    "The feasible corner points include (0,0), (4,0), (4,2), (3,3), and (0,3).",
                    "Z values are 0, 12, 16, 15, and 6 respectively.",
                    "The maximum is Z = 16 at the corner point (4,2).",
                ],
                "visual_focus": "optimization",
                "answer_line": "Constraint-based optimization compares feasible candidates and chooses the best value.",
            }

        if pattern == "algebra_roots":
            return {
                "title": "Algebraic Structure",
                "learning_goal": "Use roots, factors, or substitution to understand an algebraic expression.",
                "formula_latex": formula,
                "steps": [
                    "Identify the expression and the value or factor being tested.",
                    "Connect a factor, root, or divisor to a substitution value.",
                    "Substitute carefully and simplify the expression.",
                    "Interpret zero, remainder, or factor result from the simplified value.",
                ],
                "worked_example": [
                    "If the divisor is x - 2, then a = 2.",
                    "For p(x), calculate p(2).",
                    "If p(2) = 0, then x - 2 is a factor; otherwise p(2) is the remainder.",
                ],
                "visual_focus": "algebra",
                "answer_line": "Algebraic structure becomes clearer when factors and substitutions are connected.",
            }

        return None

    def _fallback_plan(self, query: str, context_text: str) -> dict[str, object]:
        pattern_plan = self._pattern_fallback_plan(query, context_text)
        if pattern_plan:
            return pattern_plan

        scope = f"{query} {context_text}".lower()
        visual_focus = "generic"
        pattern = self._teaching_pattern(query, context_text)
        if pattern == "geometry_measure" and "triangle" in scope:
            visual_focus = "triangle"
        elif pattern == "geometry_measure" and ("circle" in scope or "sphere" in scope):
            visual_focus = "circle"
        elif pattern == "algebra_roots" or "equation" in scope or "algebra" in scope:
            visual_focus = "algebra"

        formula_latex = self._fallback_formula(query, context_text)
        query_line = self._clip(query, 78)
        return {
            "title": "Concept Walkthrough",
            "learning_goal": f"Explain: {query_line}",
            "formula_latex": formula_latex,
            "steps": [
                "Identify the main concept and the question being asked.",
                "Name the important relationship or rule in simple words.",
                "Connect each part of the rule to the textbook context.",
                "Use a short example to check that the idea makes sense.",
            ],
            "worked_example": [
                "Start with the core idea from the question.",
                f"Use the relationship: {self._latex_to_text(formula_latex)}",
                "Explain the result in one clear sentence.",
            ],
            "visual_focus": visual_focus,
            "answer_line": "The answer follows from connecting the concept to a clear example.",
        }

    def _normalize_plan(
        self,
        raw_plan: dict | None,
        query: str,
        context_text: str,
    ) -> dict[str, object]:
        base = self._fallback_plan(query, context_text)
        if not raw_plan:
            return base

        plan = dict(base)
        for field in ("title", "learning_goal", "formula_latex", "visual_focus", "answer_line"):
            value = raw_plan.get(field)
            if isinstance(value, str) and value.strip():
                plan[field] = self._clip(value, 220)

        for field in ("steps", "worked_example"):
            value = raw_plan.get(field)
            cleaned: list[str] = []
            if isinstance(value, list):
                cleaned = [self._clip(str(item), 180) for item in value if str(item).strip()]
            elif isinstance(value, str) and value.strip():
                pieces = re.split(r"(?:\n+|•|- )", value.strip())
                cleaned = [self._clip(piece, 180) for piece in pieces if piece.strip()]
            if cleaned:
                plan[field] = cleaned[:6] if field == "steps" else cleaned[:5]

        visual = str(plan.get("visual_focus", "generic")).strip().lower()
        if visual not in {"triangle", "circle", "algebra", "identity", "network", "optimization", "generic"}:
            visual = "generic"
        plan["visual_focus"] = visual

        formula = str(plan.get("formula_latex", "")).strip()
        if not formula:
            plan["formula_latex"] = self._fallback_formula(query, context_text)

        if not plan["steps"]:
            plan["steps"] = base["steps"]
        if not plan["worked_example"]:
            plan["worked_example"] = base["worked_example"]
        return plan

    @staticmethod
    def _plan_looks_specific(plan: dict[str, object], query: str) -> bool:
        query_terms = {
            term
            for term in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{3,}", query.lower())
            if term not in {"explain", "show", "steps", "with", "what", "does", "mean"}
        }
        fields: list[str] = [
            str(plan.get("title", "")),
            str(plan.get("learning_goal", "")),
            str(plan.get("formula_latex", "")),
            str(plan.get("answer_line", "")),
        ]
        fields.extend(str(item) for item in (plan.get("steps") or []))
        fields.extend(str(item) for item in (plan.get("worked_example") or []))
        combined = " ".join(fields).lower()

        placeholder_fragments = (
            "write the key formula",
            "given values from the question",
            "compute each step",
            "step-by-step concept walkthrough",
            "answer follows from applying the formula",
            "define the key relationship",
            "concept walkthrough",
            "answer follows from connecting the concept",
        )
        if any(fragment in combined for fragment in placeholder_fragments):
            return False
        if query_terms and not any(term in combined for term in query_terms):
            return False
        return True

    def _generate_plan(self, query: str, context_text: str) -> tuple[dict[str, object], str]:
        if not self._inference.is_configured():
            return self._fallback_plan(query, context_text), "inference_unavailable_fallback"

        style_context = self._load_skill_context()
        prompt = self._plan_prompt(query=query, context_text=context_text, style_context=style_context)
        try:
            response = self._inference.chat_completions(
                [{"role": "user", "content": prompt}],
                max_tokens=min(900, max(500, self._inference.max_tokens)),
            )
            content, _reasoning = self._inference.extract_response_payload(response)
            parsed = self._extract_json_object(content)
            plan = self._normalize_plan(parsed, query, context_text)
            if self._plan_looks_specific(plan, query):
                return plan, "llm_plan"
            return self._fallback_plan(query, context_text), "plan_quality_fallback"
        except Exception:
            return self._fallback_plan(query, context_text), "plan_fallback"

    def _plan_to_markdown(
        self,
        request: PluginJobRequest,
        plan: dict[str, object],
        plan_mode: str,
    ) -> str:
        steps = plan.get("steps") or []
        worked = plan.get("worked_example") or []
        step_lines = "\n".join([f"{idx + 1}. {item}" for idx, item in enumerate(steps)])
        worked_lines = "\n".join([f"- {item}" for item in worked])
        return dedent(
            f"""
            # DeepGyan Animation Plan

            - Plugin: `{request.plugin_id}`
            - Mode: `{request.mode}`
            - Focus page: `{request.current_page}`
            - Plan source: `{plan_mode}`
            - Query: {request.query}

            ## Title
            {plan.get("title", "")}

            ## Learning Goal
            {plan.get("learning_goal", "")}

            ## Formula
            {plan.get("formula_latex", "")}

            ## Steps
            {step_lines}

            ## Worked Example
            {worked_lines}

            ## Answer Line
            {plan.get("answer_line", "")}
            """
        ).strip()

    def _template_script_from_plan(self, query: str, plan: dict[str, object]) -> str:
        title = self._clip(str(plan.get("title", "") or "DeepGyan Animation"), 70)
        learning_goal = self._wrap_text(self._clip(str(plan.get("learning_goal", "") or query), 95), 40, 3)
        formula_text = self._wrap_text(
            self._latex_to_text(str(plan.get("formula_latex", "") or "")),
            width=34,
            max_lines=3,
        )

        steps = [self._clip(str(item), 95) for item in (plan.get("steps") or []) if str(item).strip()]
        if len(steps) < 4:
            steps = [self._clip(str(item), 95) for item in self._fallback_plan(query, "").get("steps", [])]
        steps = steps[:4]

        worked = [self._clip(str(item), 95) for item in (plan.get("worked_example") or []) if str(item).strip()]
        if len(worked) < 3:
            worked = [self._clip(str(item), 95) for item in self._fallback_plan(query, "").get("worked_example", [])]
        worked = worked[:3]

        visual_focus = str(plan.get("visual_focus", "generic"))

        lines = [
            "from manim import *",
            "",
            f"class {self._scene_name}(Scene):",
            "    def construct(self):",
            "        self.camera.background_color = '#0D1326'",
            "",
            f"        title = Text({repr(title)}, font_size=40, color=BLUE_B)",
            f"        goal = Text({repr(learning_goal)}, font_size=21).scale_to_fit_width(11.2)",
            f"        formula = Text({repr(formula_text)}, font_size=23, color=GREEN_B).scale_to_fit_width(5.8)",
            "        title.to_edge(UP, buff=0.45)",
            "        goal.next_to(title, DOWN, buff=0.35)",
            "",
            "        self.play(FadeIn(title, shift=UP * 0.25), run_time=1.0)",
            "        self.play(FadeIn(goal, shift=UP * 0.2), run_time=1.2)",
            "        self.wait(0.9)",
            "",
            "        visual_group = VGroup()",
        ]

        if visual_focus == "triangle":
            lines.extend(
                [
                    "        tri = Polygon(LEFT * 2.8 + DOWN * 1.5, RIGHT * 2.8 + DOWN * 1.5, UP * 1.8, color=YELLOW)",
                    "        base_label = Text('base b', font_size=24, color=YELLOW).next_to(tri, DOWN, buff=0.2)",
                    "        h_line = DashedLine(UP * 1.8, UP * 1.8 + DOWN * 3.3, color=BLUE_B)",
                    "        h_label = Text('height h', font_size=24, color=BLUE_B).next_to(h_line, RIGHT, buff=0.2)",
                    "        visual_group = VGroup(tri, base_label, h_line, h_label).scale(0.65)",
                ]
            )
        elif visual_focus == "circle":
            lines.extend(
                [
                    "        circle = Circle(radius=2.0, color=YELLOW)",
                    "        radius = Line(ORIGIN, RIGHT * 2.0, color=BLUE_B)",
                    "        r_label = Text('r', font_size=28, color=BLUE_B).next_to(radius, UP, buff=0.2)",
                    "        visual_group = VGroup(circle, radius, r_label).scale(0.85)",
                ]
            )
        elif visual_focus == "algebra":
            lines.extend(
                [
                    "        x_box = SurroundingRectangle(Text('x', font_size=30), color=YELLOW, buff=0.25)",
                    "        eq_hint = Text('Solve for unknown', font_size=26, color=YELLOW).next_to(x_box, DOWN, buff=0.3)",
                    "        visual_group = VGroup(x_box, eq_hint)",
                ]
            )
        elif visual_focus == "identity":
            lines.extend(
                [
                    "        lhs_box = RoundedRectangle(corner_radius=0.16, width=1.65, height=0.85, color=BLUE_B)",
                    "        rhs_box = RoundedRectangle(corner_radius=0.16, width=1.65, height=0.85, color=GREEN_B).shift(RIGHT * 4.2)",
                    "        lhs_label = Text('L.H.S.', font_size=22, color=BLUE_B).move_to(lhs_box)",
                    "        rhs_label = Text('R.H.S.', font_size=22, color=GREEN_B).move_to(rhs_box)",
                    "        condition = Text('condition: A + B = 90°', font_size=20, color=YELLOW).move_to(RIGHT * 2.1 + UP * 0.75)",
                    "        equals = Arrow(lhs_box.get_right(), rhs_box.get_left(), color=YELLOW, buff=0.18)",
                    "        check = Text('check equality', font_size=20, color=YELLOW).next_to(equals, DOWN, buff=0.2)",
                    "        visual_group = VGroup(lhs_box, rhs_box, lhs_label, rhs_label, condition, equals, check)",
                ]
            )
        elif visual_focus == "network":
            lines.extend(
                [
                    "        sender = RoundedRectangle(corner_radius=0.18, width=1.9, height=1.0, color=BLUE_B)",
                    "        receiver = RoundedRectangle(corner_radius=0.18, width=1.9, height=1.0, color=GREEN_B).shift(RIGHT * 4.6)",
                    "        sender_label = Text('Device', font_size=22, color=BLUE_B).move_to(sender)",
                    "        receiver_label = Text('Server', font_size=22, color=GREEN_B).move_to(receiver)",
                    "        link = Arrow(sender.get_right(), receiver.get_left(), color=YELLOW, buff=0.18)",
                    "        packet_1 = Dot(sender.get_right() + RIGHT * 0.25, color=YELLOW)",
                    "        packet_2 = Dot(sender.get_right() + RIGHT * 0.75, color=TEAL_B)",
                    "        packet_3 = Dot(sender.get_right() + RIGHT * 1.25, color=ORANGE)",
                    "        rate_label = Text('successful data / second', font_size=18, color=YELLOW).next_to(link, UP, buff=0.25)",
                    "        visual_group = VGroup(sender, receiver, sender_label, receiver_label, link, packet_1, packet_2, packet_3, rate_label)",
                ]
            )
        elif visual_focus == "optimization":
            lines.extend(
                [
                    "        axes = Axes(x_range=[0, 7, 1], y_range=[0, 5, 1], x_length=4.8, y_length=3.4, tips=False, axis_config={'color': BLUE_B, 'stroke_width': 2})",
                    "        feasible = Polygon(axes.c2p(0, 0), axes.c2p(4, 0), axes.c2p(4, 2), axes.c2p(3, 3), axes.c2p(0, 3), color=GREEN_B, fill_color=GREEN_B, fill_opacity=0.28)",
                    "        c1 = Line(axes.c2p(1, 5), axes.c2p(6, 0), color=YELLOW)",
                    "        c2 = Line(axes.c2p(4, 0), axes.c2p(4, 4.3), color=TEAL_B)",
                    "        c3 = Line(axes.c2p(0, 3), axes.c2p(6.5, 3), color=ORANGE)",
                    "        optimum = Dot(axes.c2p(4, 2), color=RED, radius=0.09)",
                    "        opt_label = Text('best corner', font_size=18, color=RED).next_to(optimum, UP, buff=0.12)",
                    "        z_arrow = Arrow(axes.c2p(1.2, 1.0), axes.c2p(4, 2), color=RED, buff=0.08)",
                    "        feasible_label = Text('feasible region', font_size=18, color=GREEN_B).move_to(axes.c2p(2.0, 1.35))",
                    "        objective_label = Text('maximize Z', font_size=18, color=RED).next_to(z_arrow, DOWN, buff=0.1)",
                    "        visual_group = VGroup(axes, feasible, c1, c2, c3, optimum, opt_label, z_arrow, feasible_label, objective_label)",
                ]
            )
        else:
            lines.extend(
                [
                    "        helper = RoundedRectangle(corner_radius=0.2, width=5.8, height=2.2, color=YELLOW)",
                    "        helper_text = Text('Visual model', font_size=28, color=YELLOW).move_to(helper)",
                    "        visual_group = VGroup(helper, helper_text)",
                ]
            )

        lines.extend(
            [
                "        visual_group.scale_to_fit_width(4.4).to_edge(RIGHT, buff=0.65).shift(DOWN * 0.35)",
                "        self.play(Create(visual_group), run_time=1.4)",
                "        self.wait(0.9)",
                "",
                "        formula_header = Text('Key Idea', font_size=23, color=YELLOW).next_to(goal, DOWN, buff=0.28).to_edge(LEFT, buff=0.8)",
                "        formula.next_to(formula_header, DOWN, buff=0.3).to_edge(LEFT, buff=0.8)",
                "        self.play(Write(formula_header), run_time=0.8)",
                "        self.play(FadeIn(formula, shift=UP * 0.2), run_time=1.0)",
                "        self.wait(1.2)",
                "",
                "        steps_header = Text('How to Read It', font_size=23, color=TEAL_B).next_to(formula, DOWN, buff=0.28).to_edge(LEFT, buff=0.8)",
                "        self.play(Write(steps_header), run_time=0.7)",
                "",
                f"        step_text = Text({repr(self._wrap_text(steps[0], 34, 3))}, font_size=19).scale_to_fit_width(5.8)",
                "        step_text.next_to(steps_header, DOWN, buff=0.28).to_edge(LEFT, buff=0.8)",
                "        self.play(FadeIn(step_text, shift=UP * 0.12), run_time=0.9)",
                "        self.wait(1.3)",
            ]
        )

        if visual_focus == "network":
            lines.extend(
                [
                    "        for packet in [packet_1, packet_2, packet_3]:",
                    "            self.play(packet.animate.move_to(receiver.get_left() + LEFT * 0.2), run_time=0.45)",
                    "            self.play(FadeOut(packet), run_time=0.2)",
                    "        self.wait(0.6)",
                    "",
                ]
            )

        for idx, step in enumerate(steps[1:], start=2):
            lines.extend(
                [
                    f"        step_{idx} = Text({repr(self._wrap_text(step, 34, 3))}, font_size=19).scale_to_fit_width(5.8).move_to(step_text)",
                    f"        self.play(Transform(step_text, step_{idx}), run_time=1.0)",
                    "        self.wait(1.3)",
                ]
            )

        lines.extend(
            [
                "",
                "        self.play(FadeOut(step_text), FadeOut(steps_header), run_time=0.5)",
                "        worked_header = Text('Worked Example', font_size=23, color=ORANGE).next_to(formula, DOWN, buff=0.28).to_edge(LEFT, buff=0.8)",
                "        self.play(Write(worked_header), run_time=0.8)",
                "",
                f"        w1 = Text({repr(self._wrap_text(worked[0], 34, 2))}, font_size=18).scale_to_fit_width(5.8)",
                "        w1.next_to(worked_header, DOWN, buff=0.25).to_edge(LEFT, buff=0.8)",
                f"        w2 = Text({repr(self._wrap_text(worked[1], 34, 2))}, font_size=18).scale_to_fit_width(5.8).next_to(w1, DOWN, buff=0.16).to_edge(LEFT, buff=0.8)",
                f"        w3 = Text({repr(self._wrap_text(worked[2], 34, 2))}, font_size=18, color=GREEN_B).scale_to_fit_width(5.8).next_to(w2, DOWN, buff=0.16).to_edge(LEFT, buff=0.8)",
                "        self.play(FadeIn(w1, shift=UP * 0.08), run_time=0.8)",
                "        self.wait(0.8)",
                "        self.play(FadeIn(w2, shift=UP * 0.08), run_time=0.8)",
                "        self.wait(0.8)",
                "        self.play(FadeIn(w3, shift=UP * 0.08), run_time=0.8)",
                "        self.wait(1.8)",
            ]
        )
        return "\n".join(lines).strip()

    @staticmethod
    def _script_looks_valid(script: str, scene_name: str) -> bool:
        if "from manim import" not in script:
            return False
        if f"class {scene_name}(Scene)" not in script:
            return False
        if script.count("self.wait(") < 2:
            return False
        try:
            ast.parse(script)
        except SyntaxError:
            return False
        return True

    def _generate_script(self, query: str, context_text: str, plan: dict[str, object]) -> tuple[str, str]:
        if not self._inference.is_configured():
            return self._template_script_from_plan(query, plan), "template_from_plan"

        style_context = self._load_skill_context()
        prompt = self._script_prompt_from_plan(
            query=query,
            context_text=context_text,
            style_context=style_context,
            plan=plan,
        )
        try:
            response = self._inference.chat_completions(
                [{"role": "user", "content": prompt}],
                max_tokens=min(1700, max(900, self._inference.max_tokens)),
            )
            content, _reasoning = self._inference.extract_response_payload(response)
            script = self._extract_python_block(content)
            if self._script_looks_valid(script, self._scene_name):
                return script, "llm_script_from_plan"
        except Exception:
            pass
        return self._template_script_from_plan(query, plan), "template_script_from_plan"

    def _render(self, script_path: Path, media_dir: Path) -> Path:
        media_dir.mkdir(parents=True, exist_ok=True)
        manim_cli = self._resolve_manim_cli()
        command = [
            manim_cli,
            f"-q{self._quality}",
            str(script_path),
            self._scene_name,
            "-o",
            "lesson.mp4",
            "--media_dir",
            str(media_dir),
        ]
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=self._render_timeout_seconds,
            check=False,
        )
        if process.returncode != 0:
            stderr = (process.stderr or "").strip()
            stdout = (process.stdout or "").strip()
            details = stderr or stdout or "Unknown manim render failure."
            raise RuntimeError(details[:1200])

        candidates = sorted(media_dir.rglob("lesson.mp4"))
        if not candidates:
            candidates = sorted(media_dir.rglob("*.mp4"))
        if not candidates:
            raise RuntimeError("Render completed but no output video was found.")
        return candidates[-1]

    @staticmethod
    def _resolve_manim_cli() -> str:
        direct = shutil.which("manim")
        if direct:
            return direct

        sibling = Path(sys.executable).resolve().parent / "manim"
        if sibling.exists():
            return str(sibling)

        raise RuntimeError(
            "Manim CLI not found. Install manim in this environment and ensure `manim` is available in PATH."
        )

    async def run(self, request: PluginJobRequest, emit: EmitFn) -> PluginJobResult:
        await emit("planning", "Building solution blueprint from textbook context...")
        plan, plan_mode = await asyncio.to_thread(
            self._generate_plan,
            request.query,
            request.context_text,
        )
        plan_text = self._plan_to_markdown(request, plan, plan_mode)
        (request.output_dir / "plan.md").write_text(plan_text, encoding="utf-8")
        await emit("planning", f"Blueprint ready ({plan_mode}).")

        await emit("scripting", "Generating Manim scene from blueprint...")
        script, generation_mode = await asyncio.to_thread(
            self._generate_script,
            request.query,
            request.context_text,
            plan,
        )
        script_path = request.output_dir / "script.py"
        script_path.write_text(script, encoding="utf-8")
        await emit("scripting", f"Script ready ({generation_mode}).")

        await emit("rendering", "Rendering draft animation (quality=low)...")
        video_path = await asyncio.to_thread(
            self._render,
            script_path,
            request.output_dir / "media",
        )
        await emit("rendering", "Render finished.")

        return PluginJobResult(
            plan_text=plan_text,
            script_path=str(script_path.resolve()),
            video_path=str(video_path.resolve()),
        )
