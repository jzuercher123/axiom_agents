import numpy as np
from pydantic import BaseModel
from scipy.ndimage import center_of_mass

from agents.gatekeeper import VerifiedFrame


class PhysicalObservables(BaseModel):
    frame_index: int
    sha256_short: str
    sensor_name: str
    value_units: str
    bg_median: float
    bg_std: float
    peak_intensity: float
    snr: float
    centroid_x: float
    centroid_y: float
    total_flux: float
    delta_x: float | None = None
    delta_y: float | None = None


class PhysicsExtractorAgent:
    """Layer 2: Deterministic extraction of physical observables from raw tensors."""

    def __init__(self, sigma_threshold: float = 5.0):
        self.sigma_threshold = sigma_threshold
        self._prev_centroid: tuple[float, float] | None = None

    def extract(self, frame: VerifiedFrame) -> PhysicalObservables:
        img = frame.tensor.astype(np.float64)

        # Robust background estimation using median and MAD
        bg_median = float(np.median(img))
        mad = float(np.median(np.abs(img - bg_median)))
        bg_std = max(1.4826 * mad, 1e-6)  # Scale MAD to Gaussian std

        # Background-subtracted signal
        signal = img - bg_median
        peak = float(np.max(signal))
        snr = peak / bg_std

        # Mask anomaly pixels above N-sigma for sub-pixel centroid & integrated flux
        mask = signal > (self.sigma_threshold * bg_std)
        if np.any(mask):
            cy, cx = center_of_mass(np.where(mask, signal, 0.0))
            total_flux = float(np.sum(signal[mask]))
        else:
            cy, cx = np.unravel_index(np.argmax(signal), signal.shape)
            total_flux = peak

        cx, cy = float(cx), float(cy)

        # Compute displacement relative to previous anomaly
        dx = dy = None
        if self._prev_centroid is not None:
            dx = cx - self._prev_centroid[0]
            dy = cy - self._prev_centroid[1]
        self._prev_centroid = (cx, cy)

        return PhysicalObservables(
            frame_index=frame.frame_index,
            sha256_short=frame.sha256_hex[:12],
            sensor_name=frame.sensor_meta["name"],
            value_units=frame.sensor_meta["value_units"],
            bg_median=round(bg_median, 3),
            bg_std=round(bg_std, 3),
            peak_intensity=round(float(np.max(img)), 3),
            snr=round(snr, 2),
            centroid_x=round(cx, 3),
            centroid_y=round(cy, 3),
            total_flux=round(total_flux, 2),
            delta_x=round(dx, 3) if dx is not None else None,
            delta_y=round(dy, 3) if dy is not None else None,
        )