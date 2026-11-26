import json
import os
from dataclasses import dataclass
from openai import OpenAI

from agents.extractor import PhysicalObservables


@dataclass(frozen=True)
class LLMSymbolicProposal:
    name: str
    reasoning: str
    expr_x: str
    expr_y: str
    params_x: list[str]
    params_y: list[str]


class LLMHypothesisAgent:
    """Divergent reasoning engine: proposes nonlinear symbolic ansatzes when polynomials fail."""

    def __init__(self):
        # Defaults to local Ollama; override with env vars for OpenAI/Groq/OpenRouter
        api_key = os.getenv("LLM_API_KEY", os.getenv("OPENAI_API_KEY", "ollama"))
        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
        default_base_url = ollama_host if ollama_host.endswith("/v1") else f"{ollama_host}/v1"
        base_url = os.getenv("LLM_BASE_URL", default_base_url)
        self.model = os.getenv("LLM_MODEL", "qwen2.5:3b")
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def propose_nonlinear(
        self,
        history: list[PhysicalObservables],
        baseline_name: str,
        baseline_rmse: float,
    ) -> LLMSymbolicProposal | None:
        trajectory = [
            {"t": obs.frame_index, "x": obs.centroid_x, "y": obs.centroid_y}
            for obs in history
        ]

        prompt = f"""You are a theoretical physicist analyzing optical tracking telemetry.
Classical polynomial models failed (best baseline: '{baseline_name}', RMSE = {baseline_rmse} px).

Observed trajectory (t in frames, x and y in pixels):
{json.dumps(trajectory)}

Propose ONE candidate nonlinear symbolic model for x(t) and y(t) using standard math functions (sin, cos, exp, sqrt, log).
Return ONLY valid JSON matching this exact schema:
{{
  "name": "short_snake_case_name",
  "reasoning": "one sentence physical intuition",
  "expr_x": "sympy expression in t and params_x, e.g. x0 + A * cos(w * t + phi)",
  "expr_y": "sympy expression in t and params_y, e.g. y0 + B * sin(w * t + phi)",
  "params_x": ["x0", "A", "w", "phi"],
  "params_y": ["y0", "B", "w", "phi"]
}}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                response_format={"type": "json_object"},
            )
            data = json.loads(response.choices[0].message.content)
            return LLMSymbolicProposal(
                name=data["name"],
                reasoning=data["reasoning"],
                expr_x=data["expr_x"],
                expr_y=data["expr_y"],
                params_x=data["params_x"],
                params_y=data["params_y"],
            )
        except Exception as exc:
            print(f"        [llm_proposer] Failed to generate proposal: {exc}")
            return None