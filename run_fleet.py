from datetime import datetime, timezone
import json
import os
import pathlib
import zmq

from agents.extractor import PhysicalObservables, PhysicsExtractorAgent
from agents.gatekeeper import GatekeeperAgent
from agents.hypothesis import HypothesisAgent
from agents.llm_proposer import LLMHypothesisAgent, LLMSymbolicProposal
from agents.verifier import VerifierAgent


def main():
    manifest_path = pathlib.Path(__file__).resolve().parent / "config" / "fleet.yaml"

    gatekeeper = GatekeeperAgent(manifest_path)
    extractor = PhysicsExtractorAgent(sigma_threshold=5.0)
    proposer = HypothesisAgent()
    llm_proposer = LLMHypothesisAgent()
    verifier = VerifierAgent()

    history: list[PhysicalObservables] = []
    discovered_models: list[LLMSymbolicProposal] = []

    ctx = zmq.Context()
    sub = ctx.socket(zmq.SUB)
    endpoint = os.getenv("ZMQ_CORE_ENDPOINT", gatekeeper.endpoint)
    sub.connect(endpoint)
    sub.setsockopt_string(zmq.SUBSCRIBE, gatekeeper.topic)

    ui_pub_endpoint = os.getenv("ZMQ_UI_PUB_ENDPOINT", "tcp://*:5556")
    ui_pub = ctx.socket(zmq.PUB)
    ui_pub.bind(ui_pub_endpoint)

    print("[fleet] 4-Layer Agent Stack online (Gatekeeper -> Extractor -> Hypothesis -> Verifier)")
    print(f"[fleet] LLM Escalation ready via Ollama ({llm_proposer.model}) | Hypothesis Cache: ON")
    print(f"[fleet] Listening on {endpoint} | topic='{gatekeeper.topic}'")
    print(f"[fleet] Publishing verified events to UI on {ui_pub_endpoint}")
    print("[fleet] Run sda_node or mock_orbital_mode.py now...\n")

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

                # Re-evaluate any previously discovered & cached LLM models deterministically
                for cached in discovered_models:
                    cached_result = verifier.evaluate_llm_proposal(history, cached)
                    if cached_result and cached_result.bic_score < winner.bic_score:
                        winner = cached_result

                # Escalate to Ollama ONLY if both classical AND cached models fail (RMSE > 0.5 px)
                if winner.rmse_px > 0.5 and len(history) >= 6:
                    print(f"        [fleet] High residual ({winner.rmse_px} px) -> Escalating to Ollama...")
                    proposal = llm_proposer.propose_nonlinear(history, winner.name, winner.rmse_px)
                    if proposal:
                        print(f"        [llm] Proposed '{proposal.name}': {proposal.reasoning}")
                        llm_result = verifier.evaluate_llm_proposal(history, proposal)
                        if llm_result:
                            print(f"        [verifier] LLM fit: RMSE={llm_result.rmse_px} px | BIC={llm_result.bic_score}")
                            if llm_result.bic_score < winner.bic_score:
                                winner = llm_result
                                discovered_models.append(proposal)
                                print("        [verifier] LLM HYPOTHESIS VERIFIED, PROMOTED & CACHED!")

                print(
                    f"        [verifier] BEST MODEL: {winner.name} | "
                    f"RMSE={winner.rmse_px} px | BIC={winner.bic_score}\n"
                    f"                   {winner.equation_x}  |  {winner.equation_y}"
                )

                # Forward verified (non-abstained) models to axiom-sda dashboard
                if not winner.abstained:
                    ui_event = {
                        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "summary": (
                            f"Verified {winner.name} | {winner.equation_x} | {winner.equation_y} "
                            f"(RMSE={winner.rmse_px} px, BIC={winner.bic_score})"
                        ),
                        "signature_sha256": getattr(verified, "sha256_hex", obs.sha256_short),
                        "attestation_valid": True,
                    }
                    ui_pub.send_string(json.dumps(ui_event))
    except KeyboardInterrupt:
        print("\n[fleet] Shutting down.")
    finally:
        sub.close()
        ui_pub.close()
        ctx.term()


if __name__ == "__main__":
    main()