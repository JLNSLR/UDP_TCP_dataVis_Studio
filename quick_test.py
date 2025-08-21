# save as test_sender.py and run with: python test_sender.py
import socket, time, numpy as np
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
addr = ("127.0.0.1", 5005)
t = 0.0
while True:
    # 4-channel vector with some sines
    v = np.array([np.sin(t), np.cos(0.7*t), 0.5*np.sin(1.3*t), t%1], dtype=np.float32)
    sock.sendto(v.tobytes(), addr)
    t += 0.05
    time.sleep(0.01)  # 100 Hz
