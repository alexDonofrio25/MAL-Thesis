# SPDX-License-Identifier: MIT
# Copyright (c) 2020 Henrik Blidh
# Copyright (c) 2022 The Pybricks Authors

import asyncio
from bleak import BleakScanner, BleakClient
import time

UART_SERVICE_UUID = "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
UART_RX_CHAR_UUID = "6E400002-B5A3-F393-E0A9-E50E24DCCA9E"
UART_TX_CHAR_UUID = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"

# Replace this with the name of your hub if you changed
# it when installing the Pybricks firmware.
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

    async def read():
        UART_TX_CHAR_UUID = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"
        dec = await client.read_gatt_char(UART_TX_CHAR_UUID)
        data = dec.decode()
        return data

    try:
        # Connect and get services.
        await client.connect()
        await client.start_notify(UART_TX_CHAR_UUID, handle_rx)
        nus = client.services.get_service(UART_SERVICE_UUID)
        rx_char = nus.get_characteristic(UART_RX_CHAR_UUID)

        # Tell user to start program on the hub.
        print("Start the program on the hub now with the button.")
        ack = None
        time.sleep(5)
        while ack == None:
            print('Starting...')
            ack = await read()
            if ack != None:
                print('ack received')
            else:
                print('no ack')

        str = None
        print('Make Spiky look something close to it...')
        while str == None:
                str = await read()
                print('Nothing happened..')
        print ('----------------------------------')
        print ('Communication start...')


        # Send a few messages to the hub.
        for i in range(5):
            await send(client, b"fwd05")
            await asyncio.sleep(1)
            await send(client, b"turnR")
            await asyncio.sleep(1)

        # Send a message to indicate stop.
        await send(client, b"exitt")

    except Exception as e:
        # Handle exceptions.
        print(e)
    finally:
        # Disconnect when we are done.
        await client.disconnect()


# Run the main async program.
asyncio.run(main())

