import json
from types import SimpleNamespace
import zmq

from agents.hypothesis import HypothesisAgent
from agents.llm_proposer import LLMHypothesisAgent, LLMSymbolicProposal
from agents.verifier import VerifierAgent


def main():
    proposer = HypothesisAgent()
    llm_proposer = LLMHypothesisAgent()
    verifier = VerifierAgent()

    history: list[SimpleNamespace] = []
    discovered_models: list[LLMSymbolicProposal] = []

    ctx = zmq.Context()
    sub = ctx.socket(zmq.SUB)
    sub.connect("tcp://127.0.0.1:5556")
    sub.setsockopt_string(zmq.SUBSCRIBE, "obs")

    pub = ctx.socket(zmq.PUB)
    pub.bind("tcp://127.0.0.1:5557")

    print(f"[discovery] Online ({llm_proposer.model}) | SUB :5556 -> PUB :5557")

    try:
        while True:
            _, payload = sub.recv_multipart()
            obs = SimpleNamespace(**json.loads(payload.decode("utf-8")))
            history.append(obs)

            if len(history) < 3:
                continue

            candidates = proposer.propose(len(history))
            winner = verifier.evaluate(history, candidates)

            for cached in discovered_models:
                cached_res = verifier.evaluate_llm_proposal(history, cached)
                if cached_res and cached_res.bic_score < winner.bic_score:
                    winner = cached_res

            if winner.rmse_px > 0.5 and len(history) >= 6:
                print(f"[discovery] Frame #{obs.frame_index:02d} high residual ({winner.rmse_px} px) -> Escalating to Ollama...")
                proposal = llm_proposer.propose_nonlinear(history, winner.name, winner.rmse_px)
                if proposal:
                    llm_res = verifier.evaluate_llm_proposal(history, proposal)
                    if llm_res and llm_res.bic_score < winner.bic_score:
                        winner = llm_res
                        discovered_models.append(proposal)
                        print(f"[discovery] PROMOTED & CACHED: {winner.name} (RMSE={winner.rmse_px} px)")

                        event = {
                            "name": winner.name,
                            "promoted_at_frame": obs.frame_index,
                            "rmse_px": winner.rmse_px,
                            "bic_score": winner.bic_score,
                            "equation_x": winner.equation_x,
                            "equation_y": winner.equation_y,
                            "expr_x": winner.expr_x,
                            "expr_y": winner.expr_y,
                            "fitted_params_x": winner.fitted_params_x,
                            "fitted_params_y": winner.fitted_params_y,
                            "reasoning": proposal.reasoning,
                        }
                        pub.send_multipart([b"model", json.dumps(event).encode("utf-8")])
    except KeyboardInterrupt:
        pass
    finally:
        sub.close()
        pub.close()
        ctx.term()


if __name__ == "__main__":
    main()