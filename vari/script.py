# SPDX-License-Identifier: MIT
# Copyright (c) 2020 Henrik Blidh
# Copyright (c) 2022 The Pybricks Authors

import asyncio
from bleak import BleakScanner, BleakClient
import numpy as np
from pybricksdev import ble

UART_SERVICE_UUID = "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
UART_RX_CHAR_UUID = "6E400002-B5A3-F393-E0A9-E50E24DCCA9E"
UART_TX_CHAR_UUID = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"

# the name of the hub
HUB_NAME = "Spiky"


def hub_filter(device, ad):
    return device.name and device.name.lower() == HUB_NAME.lower()


def handle_disconnect(_):
    print("Hub was disconnected.")


def handle_rx(_, data: bytearray):
    print("Received:", data)


async def main():
    # Find the device and initialize client.
    device = await BleakScanner.find_device_by_filter(hub_filter)
    client = BleakClient(device, disconnected_callback=handle_disconnect)

    # Shorthand for sending some data to the hub.
    async def send(client, data):
        await client.write_gatt_char(rx_char, data)

    try:
        # Connect and get services.
        await client.connect()
        await client.start_notify(UART_TX_CHAR_UUID, handle_rx)
        nus = client.services.get_service(UART_SERVICE_UUID)
        rx_char = nus.get_characteristic(UART_RX_CHAR_UUID)

        # Tell user to start program on the hub.
        print("Start the program on the hub now with the button.")
        await asyncio.sleep(1)

        # Send a few messages to the hub.
        list = np.array([0,1,2])
        print(list)

        for i in list:
            print(i)
            keyboard_input = input("Please enter a string:\n")
            mod_input = bytearray(keyboard_input, encoding='utf-8')
            await asyncio.sleep(1)
            await send(client, mod_input)
            await asyncio.sleep(2)
            await send(client, mod_input)
            await asyncio.sleep(1)

        # Send a message to indicate stop.
        await send(client, b"stp")
        await asyncio.sleep(1)
        await send(client, b"bye")
        await asyncio.sleep(1)


    except Exception as e:
        # Handle exceptions.
        print(e)
    finally:
        # Disconnect when we are done.
        await client.disconnect()


# Run the main async program.
asyncio.run(main())

