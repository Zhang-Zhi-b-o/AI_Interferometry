"""Capture micrometer data - press button now!"""
import msvcrt, time, sys

print("Focus this window, then press/hold the micrometer button!")
print("Starting in 3...")
time.sleep(1)
print("2...")
time.sleep(1)
print("1...")
time.sleep(1)
print("GO! Press button now! (10s window)")
sys.stdout.flush()

t0 = time.time()
buf = b""
while time.time() - t0 < 10:
    if msvcrt.kbhit():
        ch = msvcrt.getch()
        buf += ch
        print(f"[{time.time()-t0:.2f}s] hex={ch.hex()} char={repr(ch)}")
        sys.stdout.flush()
    time.sleep(0.005)

if buf:
    print(f"\nReceived {len(buf)} bytes: {buf}")
    try:
        print(f"Text: {buf.decode('utf-8', errors='replace')}")
    except:
        pass
else:
    print("\nNo keyboard input received")
