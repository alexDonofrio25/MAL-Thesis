
import asyncio
from bleak import BleakScanner, BleakClient
import bleak.backends.service as bs
import numpy as np
from pybricksdev import ble
import pybricksdev.ble.lwp3.messages as msg
import pybricksdev.ble.lwp3.bytecodes as btc
import pybricksdev.ble.pybricks as pb

# variables
# Nordic UART Service setup
UART_SERVICE_UUID = "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
UART_RX_CHAR_UUID = "6E400002-B5A3-F393-E0A9-E50E24DCCA9E"
UART_TX_CHAR_UUID = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"
HUB_NAME = "Spiky" # the name of the hub

# functions
def handle_disconnect(_):
    print("Hub was disconnected.")


def handle_rx(_, data: bytearray):
    print("Received:", data)

async def bleak_client():
    device = await BleakScanner.find_device_by_name(HUB_NAME)
    client = BleakClient(device, disconnected_callback=handle_disconnect)
    print(client)
    return client



async def bleak_connection(client):
    print('client: ' + str(client))
    async with client:
        await client.connect()
        check =client.is_connected
        print('Connected:' + str(check))
        print('Client information:')
        print('Address:' + str(client.address))
        print('MTU size:' + str(client.mtu_size))
        return client

async def input_program(client):
    try:
        # Connect and get services.
        await client.connect()
        await client.start_notify(UART_TX_CHAR_UUID, handle_rx)
        nus = client.services.get_service(UART_SERVICE_UUID)
        rx_char = nus.get_characteristic(UART_RX_CHAR_UUID)

        # Shorthand for sending some data to the hub.
        async def send(client, data):
            await client.write_gatt_char(rx_char, data)

         # Tell user to start program on the hub.
        print("Start the program on the hub now with the button.")
        keyboard_input = ''

        while keyboard_input != 'exitt':
            print('Write "exit" to stop the program')
            keyboard_input = input("Please enter a string:\n")
            mod_input = bytearray(keyboard_input, encoding='utf-8')
            #await asyncio.sleep(1)
            await send(client, mod_input)
            await asyncio.sleep(3)

    except Exception as e:
        # Handle exceptions.
        print(e)
    finally:
        # Disconnect when we are done.
        await client.disconnect()

async def pybricks_connection(rx,tx,maxSize, client):
    bl = ble.BLEConnection(rx,tx,maxSize)
    await bl.connect(client)
    status = bl.connected
    print(status)





# main asyncronous function
async def main():
    cli = await bleak_client()
    await input_program(cli)





# main code runned with asyncio (asyncio should be used just one time)
asyncio.run(main())

