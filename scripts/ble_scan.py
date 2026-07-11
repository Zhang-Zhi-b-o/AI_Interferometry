"""BLE 设备扫描 + GATT 服务探索 — 找出青量螺旋测微器"""
import asyncio
import sys
import argparse
from bleak import BleakScanner, BleakClient

sys.stdout.reconfigure(encoding='utf-8')

# 常见测量设备/串口 BLE 服务 UUID
KNOWN_UUIDS = {
    "0000180a-0000-1000-8000-00805f9b34fb": "Device Information",
    "00001800-0000-1000-8000-00805f9b34fb": "Generic Access",
    "00001801-0000-1000-8000-00805f9b34fb": "Generic Attribute",
    "00001812-0000-1000-8000-00805f9b34fb": "HID",
    "0000180f-0000-1000-8000-00805f9b34fb": "Battery",
    "0000fff0-0000-1000-8000-00805f9b34fb": "Custom (常见串口透传)",
}

async def scan_all():
    print("[SCAN] 扫描 BLE 设备 (8秒)...\n")
    devices = await BleakScanner.discover(timeout=8.0, return_adv=True)
    for i, (addr, (d, adv)) in enumerate(devices.items()):
        rssi = getattr(adv, 'rssi', '?')
        local = adv.local_name or ''
        name_str = d.name or local or '(no name)'
        rssi_str = str(rssi) if rssi != '?' else '?'
        mfg = ''
        if hasattr(adv, 'manufacturer_data') and adv.manufacturer_data:
            for k, v in adv.manufacturer_data.items():
                mfg += f' mfg_id=0x{k:04X} data={v.hex()[:20]}'
        print(f"  [{i}] {name_str:30s}  RSSI={rssi_str:>4s}  {addr}{mfg}")
    return devices

async def explore_device(address: str):
    """尝试连接并列出所有服务和特征值"""
    print(f"\n[CONNECT] 尝试连接 {address} ...")
    try:
        async with BleakClient(address, timeout=8.0) as client:
            print(f"  连接成功!")
            for service in client.services:
                desc = KNOWN_UUIDS.get(service.uuid, service.description or '')
                print(f"\n  Service: {service.uuid}")
                if desc:
                    print(f"    -> {desc}")
                for char in service.characteristics:
                    props = char.properties
                    print(f"    Char: {char.uuid}  props={props}")
                    if desc2 := char.description:
                        print(f"      desc: {desc2}")
                    # 尝试读取
                    if "read" in props:
                        try:
                            val = await client.read_gatt_char(char.uuid)
                            # 尝试解析为字符串
                            try:
                                text = val.decode('ascii').strip()
                                print(f"      [READ] (ascii) = '{text}'")
                            except:
                                print(f"      [READ] (hex) = {val.hex()}")
                        except Exception as e:
                            print(f"      [READ] error: {e}")
            return True
    except Exception as e:
        print(f"  连接失败: {e}")
        return False

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device", type=int, help="设备编号，不指定则扫描所有")
    parser.add_argument("-s", "--scan-only", action="store_true", help="仅扫描")
    args = parser.parse_args()

    devices = await scan_all()
    if not devices:
        print("[FAIL] 未发现 BLE 设备")
        return

    if args.scan_only:
        return

    if args.device is not None:
        # 连接指定设备
        addrs = list(devices.keys())
        if args.device >= len(addrs):
            print(f"设备编号超出范围 (0-{len(addrs)-1})")
            return
        await explore_device(addrs[args.device])
    else:
        # 尝试所有设备（但只深入探索有名字的，节省时间）
        print("\n" + "="*60)
        print("尝试探索所有有名称的设备...")
        print("="*60)
        for addr, (d, adv) in devices.items():
            local = adv.local_name or ''
            display = d.name or local or ''
            if display:
                await explore_device(addr)
                print("\n---")

if __name__ == "__main__":
    asyncio.run(main())
