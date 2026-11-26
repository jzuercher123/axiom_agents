import time
import numpy as np
import zmq

ctx = zmq.Context()
pub = ctx.socket(zmq.PUB)
pub.bind("tcp://127.0.0.1:5555")
time.sleep(0.5)  # Allow subscriber to connect

print("[mock_orbital] Publishing 10 frames of an orbiting anomaly...")
for t in range(1, 11):
    img = np.random.normal(20, 4, (64, 64)).clip(0, 255).astype(np.uint8)
    cx = int(round(32 + 14 * np.cos(0.6 * t)))
    cy = int(round(32 + 14 * np.sin(0.6 * t)))
    img[cy - 2 : cy + 3, cx - 2 : cx + 3] = 240  # Bright point source

    pub.send_multipart([b"sda.node_01", bytes(176), img.tobytes()])
    time.sleep(0.2)

print("[mock_orbital] Done.")