# Axiom Fleet Discovery Report
**Timestamp (UTC):** 2026-11-19T04:15:17.779625+00:00  
**Model:** `LLM:orbital_motion`  
**Verification Status:** **FALSIFIED**

## Physical Intuition
> This model assumes the trajectory follows an elliptical orbit, common in celestial mechanics.

## Discovered Equations of Motion
* **x(t) = x0 + A * cos(w * t + phi)  {'x0': 32.001, 'A': -14.19, 'w': 0.593, 'phi': 3.142}**
* **y(t) = y0 + B * sin(w * t + phi)  {'y0': 32.309, 'B': -13.676, 'w': 0.604, 'phi': -3.142}**

## Statistical & Out-of-Sample Validation
| Metric | Value |
| :--- | :--- |
| **In-Sample Training RMSE** | `0.2983 px` |
| **Out-of-Sample Prediction RMSE** | `4.458 px` |
| **Held-Out Test Frames** | `[11, 12, 13]` |

## Provenance
All underlying sensor frames were SHA-256 verified by `GatekeeperAgent` and archived to `archive/telemetry.db` prior to symbolic modeling.
