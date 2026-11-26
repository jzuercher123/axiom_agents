import json
import pathlib
import zmq

from agents.reporter import ScientificReporterAgent


def main():
    root = pathlib.Path(__file__).resolve().parent.parent
    reporter = ScientificReporterAgent(root / "reports")

    ctx = zmq.Context()
    sub = ctx.socket(zmq.SUB)
    sub.connect("tcp://127.0.0.1:5558")
    sub.setsockopt_string(zmq.SUBSCRIBE, "verdict")

    print("[reporter] Online | Listening on :5558 -> Writing to reports/")

    try:
        while True:
            _, payload = sub.recv_multipart()
            verdict = json.loads(payload.decode("utf-8"))
            report_path = reporter.write_report(verdict)
            print(f"[reporter] Scientific report saved -> {report_path.name}")
    except KeyboardInterrupt:
        pass
    finally:
        sub.close()
        ctx.term()


if __name__ == "__main__":
    main()