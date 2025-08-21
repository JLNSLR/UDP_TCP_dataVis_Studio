import socket, time, numpy as np
IP, PORT, RATE_HZ, N_CH = "127.0.0.1", 5005, 200, 22
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
t, dt = 0.0, 1.0/RATE_HZ
freqs = 0.2 + 0.05*np.arange(N_CH)
while True:
    v = np.sin(2*np.pi*freqs*t).astype(np.float32)
    sock.sendto(v.tobytes(), (IP, PORT))
    t += dt
    time.sleep(dt)
