from dataclasses import dataclass
import numpy as np
import sympy as sp
from scipy.optimize import curve_fit, differential_evolution

from agents.extractor import PhysicalObservables
from agents.hypothesis import SymbolicHypothesis
from agents.llm_proposer import LLMSymbolicProposal


@dataclass(frozen=True)
class IdentifiabilityReport:
    is_identifiable: bool
    rank: int
    expected_rank: int
    condition_number: float
    reason: str = ""


@dataclass(frozen=True)
class VerifiedModel:
    name: str
    equation_x: str
    equation_y: str
    rmse_px: float
    bic_score: float
    expr_x: str = ""
    expr_y: str = ""
    fitted_params_x: dict | None = None
    fitted_params_y: dict | None = None
    # --- Rigorous Metrology & Abstention Fields ---
    abstained: bool = False
    abstention_reason: str = ""
    condition_number: float = 0.0
    matrix_rank: int = 0
    conformal_radius_px: float | None = None


class VerifierAgent:
    """Layer 4: Deterministic identifiability gating, model fitting, and conformal UQ."""

    def __init__(
        self,
        max_condition_number: float = 1e4,
        conformal_alpha: float = 0.05,
        min_samples: int = 5,
    ):
        self.max_condition_number = max_condition_number
        self.conformal_alpha = conformal_alpha
        self.min_samples = min_samples

    def evaluate(
        self,
        history: list[PhysicalObservables],
        hypotheses: list[SymbolicHypothesis],
    ) -> VerifiedModel:
        t = np.array([obs.frame_index for obs in history], dtype=np.float64)
        x = np.array([obs.centroid_x for obs in history], dtype=np.float64)
        y = np.array([obs.centroid_y for obs in history], dtype=np.float64)
        n = len(history)

        if n < self.min_samples or np.ptp(t) == 0.0:
            return self._abstain(
                f"Insufficient temporal support (n={n}, span={np.ptp(t):.2f})"
            )

        # Normalize time to [-1, 1] so raw frame indices don't inflate condition number
        t_norm = 2.0 * (t - t.min()) / np.ptp(t) - 1.0

        best: VerifiedModel | None = None
        last_reject_reason = "No identifiable hypothesis"

        for hyp in hypotheses:
            # 1. Boullé et al. (2026) one-shot identifiability gate on recovery matrix
            theta = np.vander(t_norm, N=hyp.degree + 1, increasing=False)
            id_rep = self._check_matrix_identifiability(theta)
            if not id_rep.is_identifiable:
                last_reject_reason = f"{hyp.name}: {id_rep.reason}"
                continue

            cx = np.polyfit(t, x, hyp.degree)
            cy = np.polyfit(t, y, hyp.degree)

            pred_x = np.polyval(cx, t)
            pred_y = np.polyval(cy, t)

            radial_err = np.sqrt((x - pred_x) ** 2 + (y - pred_y) ** 2)
            mse = float(np.mean(radial_err**2))
            rmse = float(np.sqrt(mse))
            bic = n * np.log(max(mse, 1e-12)) + hyp.num_params * np.log(max(n, 1))
            conf_radius = self._conformal_quantile(radial_err, self.conformal_alpha)

            result = VerifiedModel(
                name=hyp.name,
                equation_x=self._format_poly("x(t)", cx),
                equation_y=self._format_poly("y(t)", cy),
                rmse_px=round(rmse, 4),
                bic_score=round(float(bic), 3),
                abstained=False,
                condition_number=round(id_rep.condition_number, 2),
                matrix_rank=id_rep.rank,
                conformal_radius_px=round(conf_radius, 4),
            )

            if best is None or result.bic_score < best.bic_score:
                best = result

        if best is None:
            return self._abstain(last_reject_reason)

        return best

    def evaluate_llm_proposal(
        self,
        history: list[PhysicalObservables],
        proposal: LLMSymbolicProposal,
    ) -> VerifiedModel | None:
        t = np.array([obs.frame_index for obs in history], dtype=np.float64)
        x = np.array([obs.centroid_x for obs in history], dtype=np.float64)
        y = np.array([obs.centroid_y for obs in history], dtype=np.float64)
        n = len(history)

        if n < self.min_samples:
            print(f"        [verifier] ABSTAIN LLM equation (n={n} < {self.min_samples})")
            return None

        def param_bounds(param_names: list[str]) -> list[tuple[float, float]]:
            bounds = []
            for p in param_names:
                pl = p.lower()
                if pl.startswith(("w", "omega", "freq")):
                    bounds.append((0.0, float(np.pi)))  # Nyquist limit
                elif pl.startswith(("phi", "phase")):
                    bounds.append((-float(np.pi), float(np.pi)))
                else:
                    bounds.append((-100.0, 100.0))
            return bounds

        def fit_1d(expr_str: str, param_names: list[str], target: np.ndarray):
            t_sym = sp.Symbol("t")
            p_syms = [sp.Symbol(p) for p in param_names]
            expr = sp.sympify(expr_str)
            func = sp.lambdify([t_sym, *p_syms], expr, "numpy")

            def loss(params):
                pred = func(t, *params)
                return float(np.mean((target - pred) ** 2))

            bounds = param_bounds(param_names)
            res = differential_evolution(loss, bounds, seed=42, maxiter=200, tol=1e-4)
            try:
                lower = [b[0] for b in bounds]
                upper = [b[1] for b in bounds]
                popt, _ = curve_fit(
                    func, t, target, p0=res.x, bounds=(lower, upper), maxfev=5000
                )
            except Exception:
                popt = res.x

            # Local identifiability check via exact SymPy sensitivity Jacobian J_ij = df(t_i)/dp_j
            jac_cols = []
            for p_sym in p_syms:
                df_dp = sp.lambdify([t_sym, *p_syms], sp.diff(expr, p_sym), "numpy")
                col = np.broadcast_to(np.asarray(df_dp(t, *popt), dtype=np.float64), (n,))
                jac_cols.append(col)
            jac = np.column_stack(jac_cols) if jac_cols else np.zeros((n, 0))
            id_rep = self._check_matrix_identifiability(jac)

            return popt, func(t, *popt), id_rep

        try:
            popt_x, pred_x, id_x = fit_1d(proposal.expr_x, proposal.params_x, x)
            popt_y, pred_y, id_y = fit_1d(proposal.expr_y, proposal.params_y, y)

            if not id_x.is_identifiable or not id_y.is_identifiable:
                reason = id_x.reason if not id_x.is_identifiable else id_y.reason
                print(f"        [verifier] ABSTAIN LLM:{proposal.name} ({reason})")
                return None

            radial_err = np.sqrt((x - pred_x) ** 2 + (y - pred_y) ** 2)
            mse = float(np.mean(radial_err**2))
            rmse = float(np.sqrt(mse))
            num_params = len(proposal.params_x) + len(proposal.params_y)
            bic = n * np.log(max(mse, 1e-12)) + num_params * np.log(max(n, 1))
            conf_radius = self._conformal_quantile(radial_err, self.conformal_alpha)

            clean_x = {k: round(float(v), 3) for k, v in zip(proposal.params_x, popt_x)}
            clean_y = {k: round(float(v), 3) for k, v in zip(proposal.params_y, popt_y)}

            return VerifiedModel(
                name=f"LLM:{proposal.name}",
                equation_x=f"x(t) = {proposal.expr_x}  {clean_x}",
                equation_y=f"y(t) = {proposal.expr_y}  {clean_y}",
                rmse_px=round(rmse, 4),
                bic_score=round(float(bic), 3),
                expr_x=proposal.expr_x,
                expr_y=proposal.expr_y,
                fitted_params_x=clean_x,
                fitted_params_y=clean_y,
                abstained=False,
                condition_number=round(max(id_x.condition_number, id_y.condition_number), 2),
                matrix_rank=min(id_x.rank, id_y.rank),
                conformal_radius_px=round(conf_radius, 4),
            )
        except Exception as exc:
            print(f"        [verifier] REJECTED LLM equation ({exc})")
            return None

    def _check_matrix_identifiability(self, matrix: np.ndarray) -> IdentifiabilityReport:
        """Column-normalized rank & condition-number gate (Boullé et al., PNAS 2026)."""
        expected_rank = matrix.shape[1]
        if expected_rank == 0 or matrix.shape[0] < expected_rank:
            return IdentifiabilityReport(
                False, 0, expected_rank, float("inf"), "ABSTAIN: UNDERDETERMINED"
            )

        if not np.all(np.isfinite(matrix)):
            return IdentifiabilityReport(
                False, 0, expected_rank, float("inf"), "ABSTAIN: NON_FINITE_MATRIX"
            )

        # Normalize columns to unit L2 norm so mixed physical scales don't skew kappa
        col_norms = np.linalg.norm(matrix, axis=0)
        if np.any(col_norms < 1e-12):
            return IdentifiabilityReport(
                False, 0, expected_rank, float("inf"), "ABSTAIN: ZERO_SENSITIVITY_PARAM"
            )
        normed = matrix / col_norms

        rank = int(np.linalg.matrix_rank(normed))
        if rank < expected_rank:
            return IdentifiabilityReport(
                False,
                rank,
                expected_rank,
                float("inf"),
                f"ABSTAIN: RANK_DEFICIENT ({rank}/{expected_rank})",
            )

        kappa = float(np.linalg.cond(normed))
        if kappa > self.max_condition_number:
            return IdentifiabilityReport(
                False,
                rank,
                expected_rank,
                kappa,
                f"ABSTAIN: ILL_CONDITIONED (kappa={kappa:.1e} > {self.max_condition_number:.1e})",
            )

        return IdentifiabilityReport(True, rank, expected_rank, kappa)

    @staticmethod
    def _conformal_quantile(residuals: np.ndarray, alpha: float) -> float:
        """Finite-sample distribution-free (1 - alpha) conformal error bound."""
        n = len(residuals)
        if n == 0:
            return 0.0
        q_level = min(np.ceil((n + 1) * (1.0 - alpha)) / n, 1.0)
        return float(np.quantile(residuals, q_level, method="higher"))

    @staticmethod
    def _abstain(reason: str) -> VerifiedModel:
        return VerifiedModel(
            name="ABSTAIN",
            equation_x="N/A",
            equation_y="N/A",
            rmse_px=float("nan"),
            bic_score=float("inf"),
            abstained=True,
            abstention_reason=reason,
            condition_number=float("inf"),
            matrix_rank=0,
            conformal_radius_px=None,
        )

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