from dataclasses import dataclass
import numpy as np
import sympy as sp


@dataclass(frozen=True)
class FalsificationVerdict:
    model_name: str
    status: str  # "CONFIRMED" or "FALSIFIED"
    train_rmse: float
    oos_rmse: float
    tested_frames: list[int]
    equation_x: str
    equation_y: str
    reasoning: str


class AdversarialSkepticAgent:
    """Tier 4: Tests promoted symbolic models against unseen future frames."""

    def __init__(self, max_oos_rmse: float = 1.0):
        self.max_oos_rmse = max_oos_rmse

    def test_model(self, model_event: dict, future_obs: list[dict]) -> FalsificationVerdict:
        t = np.array([o["frame_index"] for o in future_obs], dtype=np.float64)
        x_true = np.array([o["centroid_x"] for o in future_obs], dtype=np.float64)
        y_true = np.array([o["centroid_y"] for o in future_obs], dtype=np.float64)

        try:
            t_sym = sp.Symbol("t")

            # Reconstruct x(t) with locked parameters from training
            px_names = list(model_event["fitted_params_x"].keys())
            px_vals = [model_event["fitted_params_x"][k] for k in px_names]
            fx = sp.lambdify([t_sym, *[sp.Symbol(p) for p in px_names]], sp.sympify(model_event["expr_x"]), "numpy")
            x_pred = fx(t, *px_vals)

            # Reconstruct y(t) with locked parameters from training
            py_names = list(model_event["fitted_params_y"].keys())
            py_vals = [model_event["fitted_params_y"][k] for k in py_names]
            fy = sp.lambdify([t_sym, *[sp.Symbol(p) for p in py_names]], sp.sympify(model_event["expr_y"]), "numpy")
            y_pred = fy(t, *py_vals)

            oos_rmse = float(np.sqrt(np.mean((x_true - x_pred) ** 2 + (y_true - y_pred) ** 2)))
            status = "CONFIRMED" if oos_rmse <= self.max_oos_rmse else "FALSIFIED"

        except Exception as exc:
            oos_rmse = 999.999
            status = f"FALSIFIED ({exc})"

        return FalsificationVerdict(
            model_name=model_event["name"],
            status=status,
            train_rmse=model_event["rmse_px"],
            oos_rmse=round(oos_rmse, 4),
            tested_frames=[o["frame_index"] for o in future_obs],
            equation_x=model_event["equation_x"],
            equation_y=model_event["equation_y"],
            reasoning=model_event.get("reasoning", ""),
        )