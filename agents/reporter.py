import pathlib
from datetime import datetime, timezone


class ScientificReporterAgent:
    """Tier 5: Compiles verified discoveries into human-readable Markdown reports."""

    def __init__(self, reports_dir: pathlib.Path):
        self.reports_dir = reports_dir
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def write_report(self, verdict: dict) -> pathlib.Path:
        ts = datetime.now(timezone.utc)
        filename = f"discovery_{ts.strftime('%Y%m%d_%H%M%S')}_{verdict['model_name'].replace(':', '_')}.md"
        out_path = self.reports_dir / filename

        content = f"""# Axiom Fleet Discovery Report
**Timestamp (UTC):** {ts.isoformat()}  
**Model:** `{verdict['model_name']}`  
**Verification Status:** **{verdict['status']}**

## Physical Intuition
> {verdict.get('reasoning', 'N/A')}

## Discovered Equations of Motion
* **{verdict['equation_x']}**
* **{verdict['equation_y']}**

## Statistical & Out-of-Sample Validation
| Metric | Value |
| :--- | :--- |
| **In-Sample Training RMSE** | `{verdict['train_rmse']} px` |
| **Out-of-Sample Prediction RMSE** | `{verdict['oos_rmse']} px` |
| **Held-Out Test Frames** | `{verdict['tested_frames']}` |

## Provenance
All underlying sensor frames were SHA-256 verified by `GatekeeperAgent` and archived to `archive/telemetry.db` prior to symbolic modeling.
"""
        out_path.write_text(content, encoding="utf-8")
        return out_path