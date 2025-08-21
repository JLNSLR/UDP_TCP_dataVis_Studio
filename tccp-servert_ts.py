import socket
import time
import numpy as np

HOST = "0.0.0.0"   # listen on all interfaces
PORT = 7000
RATE_HZ = 200      # frames per second
N_DATA = 21        # number of signal channels (not counting timestamp)
N_TOTAL = N_DATA + 1  # total floats per frame (signals + timestamp)

def serve():
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((HOST, PORT))
    srv.listen(1)
    print(f"TCP server on {HOST}:{PORT}  |  {RATE_HZ} Hz  |  {N_TOTAL} float32 per frame (last = timestamp [s])")

    dt = 1.0 / RATE_HZ
    freqs = 0.2 + 0.05 * np.arange(N_DATA)  # distinct sine freqs for each data channel

    while True:
        conn, addr = srv.accept()
        print(f"Client connected: {addr}")
        try:
            t0 = time.perf_counter()
            next_t = t0
            while True:
                now = time.perf_counter()
                elapsed = now - t0  # seconds since connection start (monotonic)

                # Build the frame: 21 signals + 1 timestamp (seconds)
                sigs = np.sin(2 * np.pi * freqs * elapsed).astype(np.float32)  # shape (N_DATA,)
                frame = np.empty(N_TOTAL, dtype=np.float32)
                frame[:N_DATA] = sigs
                frame[N_DATA] = elapsed  # timestamp in seconds at index 21

                conn.sendall(frame.tobytes(order="C"))

                # pace to RATE_HZ
                next_t += dt
                sleep = next_t - time.perf_counter()
                if sleep > 0:
                    time.sleep(sleep)
                else:
                    # if we fell behind, jump to now to avoid drift
                    next_t = time.perf_counter()
        except (BrokenPipeError, ConnectionResetError):
            print("Client disconnected, waiting for next client…")
        finally:
            conn.close()

if __name__ == "__main__":
    try:
        serve()
    except KeyboardInterrupt:
        print("\nServer stopped.")
