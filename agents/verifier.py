from dataclasses import dataclass
import numpy as np

from agents.extractor import PhysicalObservables
from agents.hypothesis import SymbolicHypothesis


@dataclass(frozen=True)
class VerifiedModel:
    name: str
    equation_x: str
    equation_y: str
    rmse_px: float
    bic_score: float


class VerifierAgent:
    """Layer 4: Fits candidate symbolic models and selects the Pareto-optimal winner."""

    def evaluate(
        self,
        history: list[PhysicalObservables],
        hypotheses: list[SymbolicHypothesis],
    ) -> VerifiedModel:
        t = np.array([obs.frame_index for obs in history], dtype=np.float64)
        x = np.array([obs.centroid_x for obs in history], dtype=np.float64)
        y = np.array([obs.centroid_y for obs in history], dtype=np.float64)
        n = len(history)

        best: VerifiedModel | None = None

        for hyp in hypotheses:
            cx = np.polyfit(t, x, hyp.degree)
            cy = np.polyfit(t, y, hyp.degree)

            pred_x = np.polyval(cx, t)
            pred_y = np.polyval(cy, t)

            sq_err = (x - pred_x) ** 2 + (y - pred_y) ** 2
            mse = float(np.mean(sq_err))
            rmse = float(np.sqrt(mse))

            # BIC penalizes unnecessary parameters (Occam's razor)
            bic = n * np.log(max(mse, 1e-12)) + hyp.num_params * np.log(max(n, 1))

            result = VerifiedModel(
                name=hyp.name,
                equation_x=self._format_poly("x(t)", cx),
                equation_y=self._format_poly("y(t)", cy),
                rmse_px=round(rmse, 4),
                bic_score=round(float(bic), 3),
            )

            if best is None or result.bic_score < best.bic_score:
                best = result

        assert best is not None
        return best

    @staticmethod
    def _format_poly(label: str, coeffs: np.ndarray) -> str:
        deg = len(coeffs) - 1
        terms = []
        for i, c in enumerate(coeffs):
            power = deg - i
            if power == 0:
                terms.append(f"{c:.3f}")
            elif power == 1:
                terms.append(f"{c:.3f}*t")
            else:
                terms.append(f"{c:.3f}*t^{power}")
        return f"{label} = " + " + ".join(terms)