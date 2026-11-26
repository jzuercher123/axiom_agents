import os
import subprocess
import sys
import time

WORKERS = [
    "workers.reporter_worker",
    "workers.skeptic_worker",
    "workers.discovery_worker",
    "workers.ingest_worker",
]


def main():
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    procs: list[subprocess.Popen] = []

    print("[launcher] Booting 4-process Axiom Agent Fleet...\n")
    try:
        for mod in WORKERS:
            p = subprocess.Popen([sys.executable, "-m", mod], env=env)
            procs.append(p)
            time.sleep(0.3)

        print("\n[launcher] All agents online. Fire sda_node or mock_orbital_mode.py now (Ctrl+C to stop).\n")
        for p in procs:
            p.wait()
    except KeyboardInterrupt:
        print("\n[launcher] Shutting down fleet...")
        for p in procs:
            p.terminate()
        for p in procs:
            p.wait()
        print("[launcher] Clean shutdown complete.")


if __name__ == "__main__":
    main()