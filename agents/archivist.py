import pathlib
import sqlite3
from datetime import datetime, timezone


class ArchivistAgent:
    """Tier 1: Persists cryptographically verified observables for reproducibility."""

    def __init__(self, db_path: pathlib.Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path))
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS observables (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                utc_time TEXT NOT NULL,
                frame_index INTEGER NOT NULL,
                sha256_hex TEXT NOT NULL,
                centroid_x REAL NOT NULL,
                centroid_y REAL NOT NULL,
                snr REAL NOT NULL,
                total_flux REAL NOT NULL,
                units TEXT NOT NULL
            )
            """
        )
        self.conn.commit()

    def record(self, obs_dict: dict) -> None:
        self.conn.execute(
            """
            INSERT INTO observables
            (utc_time, frame_index, sha256_hex, centroid_x, centroid_y, snr, total_flux, units)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                obs_dict["frame_index"],
                obs_dict["sha256_hex"],
                obs_dict["centroid_x"],
                obs_dict["centroid_y"],
                obs_dict["snr"],
                obs_dict["total_flux"],
                obs_dict["value_units"],
            ),
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()