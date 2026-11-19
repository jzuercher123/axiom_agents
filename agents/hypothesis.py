from dataclasses import dataclass
import sympy as sp


@dataclass(frozen=True)
class SymbolicHypothesis:
    name: str
    expr_x: sp.Expr
    expr_y: sp.Expr
    degree: int
    num_params: int


class HypothesisAgent:
    """Layer 3: Proposes candidate symbolic kinematic models over time t."""

    def __init__(self):
        self.t = sp.Symbol("t", real=True)

    def propose(self, num_samples: int) -> list[SymbolicHypothesis]:
        t = self.t
        candidates = [
            SymbolicHypothesis(
                name="stationary_source",
                expr_x=sp.Symbol("x0"),
                expr_y=sp.Symbol("y0"),
                degree=0,
                num_params=2,
            ),
        ]

        if num_samples >= 2:
            candidates.append(
                SymbolicHypothesis(
                    name="constant_velocity",
                    expr_x=sp.Symbol("x0") + sp.Symbol("vx") * t,
                    expr_y=sp.Symbol("y0") + sp.Symbol("vy") * t,
                    degree=1,
                    num_params=4,
                )
            )

        if num_samples >= 4:
            candidates.append(
                SymbolicHypothesis(
                    name="constant_acceleration",
                    expr_x=(
                        sp.Symbol("x0")
                        + sp.Symbol("vx") * t
                        + sp.Rational(1, 2) * sp.Symbol("ax") * t**2
                    ),
                    expr_y=(
                        sp.Symbol("y0")
                        + sp.Symbol("vy") * t
                        + sp.Rational(1, 2) * sp.Symbol("ay") * t**2
                    ),
                    degree=2,
                    num_params=6,
                )
            )

        return candidates