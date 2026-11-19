import hashlib
import pathlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import yaml

DTYPE_MAP = {
    "UInt8": np.uint8,
    "UInt16": np.uint16,
    "Int16": np.int16,
    "Int32": np.int32,
    "Float32": np.float32,
    "Float64": np.float64,
}


@dataclass(frozen=True)
class VerifiedFrame:
    topic: str
    frame_index: int
    sha256_hex: str
    header_bytes: bytes
    tensor: np.ndarray
    sensor_meta: dict[str, Any]


class GatekeeperAgent:
    """Layer 1: Zero-trust cryptographic & schema validation."""

    EXPECTED_HEADER_SIZE = 176

    def __init__(self, manifest_path: pathlib.Path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            self.manifest = yaml.safe_load(f)

        self.endpoint: str = self.manifest["node"]["transport"]["endpoint"]
        self.topic: str = self.manifest["node"]["transport"]["topic"]
        self.sensor_cfg: dict[str, Any] = self.manifest["sensors"][0]
        self.shape: tuple[int, ...] = tuple(self.sensor_cfg["shape"])
        self.dtype = DTYPE_MAP[self.sensor_cfg["dtype"]]
        self._frame_counter = 0

    def process(self, parts: list[bytes]) -> VerifiedFrame | None:
        if len(parts) != 3:
            print(f"[gatekeeper] REJECTED: Expected 3 ZMQ parts, got {len(parts)}")
            return None

        raw_topic, raw_header, raw_payload = parts

        # 1. Validate binary WireHeader size
        if len(raw_header) != self.EXPECTED_HEADER_SIZE:
            print(
                f"[gatekeeper] REJECTED: Header size {len(raw_header)}B "
                f"!= expected {self.EXPECTED_HEADER_SIZE}B"
            )
            return None

        # 2. Compute SHA-256 payload digest (strict C++ header-hash match deferred to v0.2)
        payload_digest = hashlib.sha256(raw_payload).digest()

        # 3. Reconstruct zero-copy NumPy tensor
        try:
            tensor = np.frombuffer(raw_payload, dtype=self.dtype).reshape(self.shape)
        except ValueError as exc:
            print(f"[gatekeeper] REJECTED: Tensor reshape failed ({exc})")
            return None

        self._frame_counter += 1
        return VerifiedFrame(
            topic=raw_topic.decode("utf-8"),
            frame_index=self._frame_counter,
            sha256_hex=payload_digest.hex(),
            header_bytes=raw_header,
            tensor=tensor,
            sensor_meta=self.sensor_cfg,
        )