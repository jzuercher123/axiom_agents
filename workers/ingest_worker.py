import json
import pathlib
import zmq

from agents.archivist import ArchivistAgent
from agents.extractor import PhysicsExtractorAgent
from agents.gatekeeper import GatekeeperAgent


def main():
    root = pathlib.Path(__file__).resolve().parent.parent
    gatekeeper = GatekeeperAgent(root / "config" / "fleet.yaml")
    extractor = PhysicsExtractorAgent(sigma_threshold=5.0)
    archivist = ArchivistAgent(root / "archive" / "telemetry.db")

    ctx = zmq.Context()
    sub = ctx.socket(zmq.SUB)
    sub.connect(gatekeeper.endpoint)
    sub.setsockopt_string(zmq.SUBSCRIBE, gatekeeper.topic)

    pub = ctx.socket(zmq.PUB)
    pub.bind("tcp://127.0.0.1:5556")

    print(f"[ingest] Online | SUB {gatekeeper.endpoint} -> PUB tcp://127.0.0.1:5556")

    try:
        while True:
            parts = sub.recv_multipart()
            verified = gatekeeper.process(parts)
            if verified is None:
                continue

            obs = extractor.extract(verified)
            obs_dict = {
                "frame_index": obs.frame_index,
                "sha256_hex": verified.sha256_hex,
                "sha256_short": obs.sha256_short,
                "centroid_x": obs.centroid_x,
                "centroid_y": obs.centroid_y,
                "snr": obs.snr,
                "total_flux": obs.total_flux,
                "value_units": obs.value_units,
            }

            archivist.record(obs_dict)
            pub.send_multipart([b"obs", json.dumps(obs_dict).encode("utf-8")])
            print(
                f"[ingest] #{obs.frame_index:02d} | sha256={obs.sha256_short}.. | "
                f"centroid=({obs.centroid_x:5.2f}, {obs.centroid_y:5.2f}) | SNR={obs.snr:5.2f}"
            )
    except KeyboardInterrupt:
        pass
    finally:
        archivist.close()
        sub.close()
        pub.close()
        ctx.term()


if __name__ == "__main__":
    main()