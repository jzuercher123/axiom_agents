from dataclasses import asdict
import json
import zmq

from agents.skeptic import AdversarialSkepticAgent


def main():
    skeptic = AdversarialSkepticAgent(max_oos_rmse=1.0)
    pending_tests: list[dict] = []
    held_out_window = 3

    ctx = zmq.Context()
    sub = ctx.socket(zmq.SUB)
    sub.connect("tcp://127.0.0.1:5556")
    sub.connect("tcp://127.0.0.1:5557")
    sub.setsockopt_string(zmq.SUBSCRIBE, "obs")
    sub.setsockopt_string(zmq.SUBSCRIBE, "model")

    pub = ctx.socket(zmq.PUB)
    pub.bind("tcp://127.0.0.1:5558")

    print("[skeptic] Online | Monitoring :5556 & :5557 -> PUB :5558")

    try:
        while True:
            topic, payload = sub.recv_multipart()
            data = json.loads(payload.decode("utf-8"))

            if topic == b"model":
                print(f"[skeptic] Queued '{data['name']}' for {held_out_window}-frame out-of-sample test...")
                pending_tests.append({"event": data, "future_obs": []})

            elif topic == b"obs":
                remaining = []
                for test in pending_tests:
                    if data["frame_index"] > test["event"]["promoted_at_frame"]:
                        test["future_obs"].append(data)

                    if len(test["future_obs"]) >= held_out_window:
                        verdict = skeptic.test_model(test["event"], test["future_obs"])
                        print(
                            f"[skeptic] VERDICT: {verdict.status} | {verdict.model_name} | "
                            f"Train RMSE={verdict.train_rmse} px -> OOS RMSE={verdict.oos_rmse} px"
                        )
                        pub.send_multipart([b"verdict", json.dumps(asdict(verdict)).encode("utf-8")])
                    else:
                        remaining.append(test)
                pending_tests = remaining
    except KeyboardInterrupt:
        pass
    finally:
        sub.close()
        pub.close()
        ctx.term()


if __name__ == "__main__":
    main()