import pathlib
import zmq

from agents.extractor import PhysicalObservables, PhysicsExtractorAgent
from agents.gatekeeper import GatekeeperAgent
from agents.hypothesis import HypothesisAgent
from agents.verifier import VerifierAgent


def main():
    manifest_path = pathlib.Path(__file__).resolve().parent / "config" / "fleet.yaml"

    gatekeeper = GatekeeperAgent(manifest_path)
    extractor = PhysicsExtractorAgent(sigma_threshold=5.0)
    proposer = HypothesisAgent()
    verifier = VerifierAgent()

    history: list[PhysicalObservables] = []

    ctx = zmq.Context()
    sub = ctx.socket(zmq.SUB)
    sub.connect(gatekeeper.endpoint)
    sub.setsockopt_string(zmq.SUBSCRIBE, gatekeeper.topic)

    print("[fleet] 4-Layer Agent Stack online (Gatekeeper -> Extractor -> Hypothesis -> Verifier)")
    print(f"[fleet] Listening on {gatekeeper.endpoint} | topic='{gatekeeper.topic}'")
    print("[fleet] Run sda_node from CLion now...\n")

    try:
        while True:
            parts = sub.recv_multipart()

            # Layer 1: Zero-trust ingest
            verified = gatekeeper.process(parts)
            if verified is None:
                continue

            # Layer 2: Deterministic physics extraction
            obs = extractor.extract(verified)
            history.append(obs)

            print(
                f"[fleet] #{obs.frame_index:02d} | sha256={obs.sha256_short}.. | "
                f"centroid=({obs.centroid_x:5.2f}, {obs.centroid_y:5.2f}) | "
                f"SNR={obs.snr:6.2f} | flux={obs.total_flux:7.1f} {obs.value_units}"
            )

            # Layers 3 & 4: Symbolic hypothesis proposal + BIC verification
            if len(history) >= 3:
                candidates = proposer.propose(len(history))
                winner = verifier.evaluate(history, candidates)
                print(
                    f"        [verifier] BEST MODEL: {winner.name} | "
                    f"RMSE={winner.rmse_px} px | BIC={winner.bic_score}\n"
                    f"                   {winner.equation_x}  |  {winner.equation_y}"
                )
    except KeyboardInterrupt:
        print("\n[fleet] Shutting down.")
    finally:
        sub.close()
        ctx.term()


if __name__ == "__main__":
    main()