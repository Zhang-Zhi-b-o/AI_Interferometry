"""捕获数显微分管 HID 键盘输出"""
import sys
import time

# 尝试方式1: 串口
print("=== 方式1: 搜索串口 ===")
import serial.tools.list_ports
ports = list(serial.tools.list_ports.comports())
if ports:
    for p in ports:
        print(f"  找到: {p.device} - {p.description} (VID:{p.vid:04X} PID:{p.pid:04X})")
        try:
            ser = serial.Serial(p.device, 9600, timeout=0.5)
            ser.reset_input_buffer()
            print(f"  等待数据(5秒)...")
            t0 = time.time()
            while time.time() - t0 < 5:
                data = ser.read(64)
                if data:
                    print(f"  收到: {data.hex(' ')} | {data}")
                time.sleep(0.05)
            ser.close()
        except Exception as e:
            print(f"  打开失败: {e}")
else:
    print("  未找到串口设备")

# 尝试方式2: 用 raw_input 直接接收
print("\n=== 方式2: stdin 监听(5秒,请在这里点按微分管按钮) ===")
print("  请确保此终端窗口获得焦点！")
import msvcrt
t0 = time.time()
buf = []
while time.time() - t0 < 5:
    if msvcrt.kbhit():
        ch = msvcrt.getch()
        buf.append(ch)
        print(f"  [{time.time()-t0:.1f}s] raw={ch.hex()} dec={ch}")
    time.sleep(0.01)
if buf:
    full = b''.join(buf)
    print(f"  共收到 {len(buf)} 字节: {full}")
    print(f"  文本: {full.decode('utf-8', errors='replace')}")
else:
    print("  未收到任何输入")

print("\n=== 完成 ===")
