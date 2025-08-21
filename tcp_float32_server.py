# tcp_float32_server.py
import socket, time, numpy as np

HOST = "0.0.0.0"   # listen on all interfaces
PORT = 7000        # must match your YAML
RATE_HZ = 200      # frames per second
N_CH = 22          # floats per frame

def serve():
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((HOST, PORT))
    srv.listen(1)
    print(f"TCP server listening on {HOST}:{PORT} (RATE {RATE_HZ} Hz, {N_CH} ch)")

    dt = 1.0 / RATE_HZ
    freqs = 0.2 + 0.05 * np.arange(N_CH)  # distinct sine freqs per channel

    while True:
        conn, addr = srv.accept()
        print(f"Client connected: {addr}")
        try:
            t = 0.0
            while True:
                v = np.sin(2 * np.pi * freqs * t).astype(np.float32)  # shape (N_CH,)
                payload = v.tobytes(order="C")  # float32 LE
                conn.sendall(payload)
                t += dt
                time.sleep(dt)
        except (BrokenPipeError, ConnectionResetError):
            print("Client disconnected, waiting for next client…")
        finally:
            conn.close()

if __name__ == "__main__":
    try:
        serve()
    except KeyboardInterrupt:
        print("\nServer stopped.")
