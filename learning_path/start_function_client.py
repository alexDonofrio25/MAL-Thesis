from HubConnection import HubConnection
from bleak import BleakScanner, BleakClient
import asyncio
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


async def handle_rx(_, data: bytearray):
    print("Received:", data)




async def main():
     # Find the device and initialize client.
    device = await BleakScanner.find_device_by_filter(hub_filter)
    client = BleakClient(device, disconnected_callback=handle_disconnect)

    # Shorthand for sending some data to the hub.
    async def send(client, data):
        await client.write_gatt_char(rx_char, data)

    async def read(client):
        dec = await client.read_gatt_char(UART_TX_CHAR_UUID)
        data = dec.decode()
        return data

    try:
        #bleHub = HubConnection('Spiky')
        #await bleHub.client_discovery()
        #await bleHub.connect()

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
            ack = await read(client)
            if ack != None:
                print('ack update:' + ack)
            else:
                print('no ack')

        # wait until an event occurs and the hub can start receiving data
        str = None
        while str == None:
            print('Make Spiky look something close to it...')
            try:
                str = await read(client)
            except Exception as e:
                print ('Nothing happened..')
            finally:
                time.sleep(2)
        print ('----------------------------------')
        print ('Communication start...')

        #start notifying data
        await send(client, b'fwd05')
        time.sleep(3)
        x = None
        while x == None:
            x = ''
    except Exception as e:
        # Handle exceptions.
        print(e)
    finally:
        # Disconnect when we are done.
        await client.disconnect()

    #while bleHub.output == None:
     #   time.sleep(3)
      #  print(bleHub.output)
    #if bleHub.output == 'start':
     #   bleHub.send(b'fwd05')
asyncio.run(main())