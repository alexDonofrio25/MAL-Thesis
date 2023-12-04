import os
import numpy as np
import asyncio
from bleak import BleakScanner, BleakClient


# Nordic UART Service setup
UART_SERVICE_UUID = "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
UART_RX_CHAR_UUID = "6E400002-B5A3-F393-E0A9-E50E24DCCA9E"
UART_TX_CHAR_UUID = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"

# Hub's name
HUB_NAME = "Spiky"


def hub_filter(device, ad):
    return device.name and device.name.lower() == HUB_NAME.lower()
# disconnection handler
def handle_disconnect(_):
    print("Hub was disconnected.")
# message received handler
def handle_rx(_, data: bytearray):
    print("Received:", data)

# create a connection with the hub
async def connectionCreation():
    # Find the device and initialize client.
    device = await BleakScanner.find_device_by_filter(hub_filter)
    client = BleakClient(device, disconnected_callback=handle_disconnect)
    try:
        # Connect and get services.
        await client.connect()
        await client.start_notify(UART_TX_CHAR_UUID, handle_rx)
        nus = client.services.get_service(UART_SERVICE_UUID)
        rx_char = nus.get_characteristic(UART_RX_CHAR_UUID)
    except Exception as e:
        # Handle exceptions.
        print(e)
    finally:
        # return the connection data
        return client



async def send(client, rx_char, data):
        await client.write_gatt_char(rx_char, data)


async def uploadCode(file):
    device = await BleakScanner.find_device_by_filter(hub_filter)
    client = BleakClient(device, disconnected_callback=handle_disconnect)
    os.system('pipx run pybricksdev run ble --name "Spiky" ' + file)
    return client


async def main_script(client):
    try:
        # Send a few messages to the hub.
            list = np.array([0,1,2])
            print(list)
            for i in list:
                await send(client, b"fwd")
                await asyncio.sleep(1)
                await send(client, b"rev")
                await asyncio.sleep(1)

            # Send a message to indicate stop.
            await send(client, b"stp")
            await send(client, b"bye")

    except Exception as e:
        # Handle exceptions.
        print(e)
    finally:
        # Disconnect when we are done.
        await client.disconnect()

#cli = asyncio.run(connectionCreation())
cli = asyncio.run(uploadCode('trial.py'))
print('hello')
asyncio.run(main_script(cli))