"""完整探查青量测微器所有 GATT 特征值 + 读取值 + HID 深度分析"""
import asyncio
import sys
sys.stdout.reconfigure(encoding='utf-8')

from winrt.windows.devices.bluetooth import BluetoothLEDevice
from winrt.windows.storage.streams import DataReader

KNOWN_CHARS = {
    "00002a00-0000-1000-8000-00805f9b34fb": "Device Name",
    "00002a01-0000-1000-8000-00805f9b34fb": "Appearance",
    "00002a04-0000-1000-8000-00805f9b34fb": "PPCP (连接参数)",
    "00002aa6-0000-1000-8000-00805f9b34fb": "Central Address Resolution",
    "00002a29-0000-1000-8000-00805f9b34fb": "Manufacturer Name String",
    "00002a50-0000-1000-8000-00805f9b34fb": "PnP ID",
    "00002a19-0000-1000-8000-00805f9b34fb": "Battery Level",
    "00002a4d-0000-1000-8000-00805f9b34fb": "HID Report",
    "00002a4e-0000-1000-8000-00805f9b34fb": "HID Report Map",
    "00002a4b-0000-1000-8000-00805f9b34fb": "HID Report Reference",
    "00002a22-0000-1000-8000-00805f9b34fb": "HID Boot Keyboard Input Report",
    "00002a32-0000-1000-8000-00805f9b34fb": "HID Boot Keyboard Output Report",
    "00002a4a-0000-1000-8000-00805f9b34fb": "HID Information",
    "00002a4c-0000-1000-8000-00805f9b34fb": "HID Control Point",
}

QLR_MAC = 0xCED9C11999E5


async def read_char_value(char):
    """读取特征值并返回 bytes"""
    try:
        read_result = await char.read_value_async()
        if read_result.status.name == "Success":
            reader = DataReader.from_buffer(read_result.value)
            data = bytearray(reader.unconsumed_buffer_length)
            reader.read_bytes(data)
            return bytes(data)
    except Exception as e:
        return f"ERROR: {e}"
    return None


async def main():
    print("=" * 60)
    print("青量 QLR-013530586 完整 GATT 探查")
    print("=" * 60)

    device = await BluetoothLEDevice.from_bluetooth_address_async(QLR_MAC)
    if not device:
        print("无法访问设备!")
        return

    print(f"设备名: {device.name}")
    print(f"连接状态: {device.connection_status}\n")

    gatt_result = await device.get_gatt_services_async()
    services = gatt_result.services

    for i in range(services.size):
        svc = services.get_at(i)
        svc_uuid = str(svc.uuid)
        print(f"\n{'='*60}")
        print(f"SERVICE [{i}]: {svc_uuid}")

        try:
            chars_result = await svc.get_characteristics_async()
            chars = chars_result.characteristics
            if chars.size == 0:
                print("  (no characteristics accessible)")
                continue

            for j in range(chars.size):
                char = chars.get_at(j)
                char_uuid = str(char.uuid)
                props = char.characteristic_properties
                props_str = str(props).replace("GattCharacteristicProperties.", "")
                char_name = KNOWN_CHARS.get(char_uuid, '')

                print(f"\n  CHAR[{j}]: {char_uuid}  [{props_str}]")
                if char_name:
                    print(f"    -> {char_name}")

                # 读取所有可读特征值
                if int(props) & 1:  # Read
                    val = await read_char_value(char)
                    if val and isinstance(val, bytes):
                        print(f"    值(hex): {val.hex()}")
                        # 尝试多种解码
                        for enc in ['ascii', 'utf-8', 'utf-16-le']:
                            try:
                                s = val.decode(enc).strip('\x00').strip()
                                if s and all(32 <= ord(c) < 127 for c in s):
                                    print(f"    值(str[{enc}]): '{s}'")
                                    break
                            except:
                                pass
                        # int 解析
                        if len(val) == 1:
                            print(f"    值(uint8): {val[0]}")
                        elif len(val) == 2:
                            print(f"    值(uint16_le): {int.from_bytes(val, 'little')}")
                        elif len(val) == 4:
                            print(f"    值(uint32_le): {int.from_bytes(val, 'little')}")
                    else:
                        print(f"    值: {val}")

                # Mark notify/subscribe capable
                if "Notify" in props_str:
                    print(f"    *** 支持 NOTIFY (可订阅推送) ***")

        except Exception as e:
            print(f"  ERROR enumerating chars: {e}")

    # === 总结 ===
    print(f"\n{'='*60}")
    print("总结")
    print("=" * 60)
    print("如果服务列表中没有任何自定义测量数据服务，")
    print("则该设备仅支持 HID 键盘模式（按按钮 = 打字），")
    print("无法通过 BLE 主动轮询测量值。")
    print("\n但可以通过 HID API 在电脑上监听键盘输入来捕获数据。")

    device.close()

if __name__ == "__main__":
    asyncio.run(main())
