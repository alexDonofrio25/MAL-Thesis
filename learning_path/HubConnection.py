from bleak import BleakScanner, BleakClient

class HubConnection():

    def __init__(self, hub_name):
        self.UART_SERVICE_UUID = "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
        self.UART_RX_CHAR_UUID = "6E400002-B5A3-F393-E0A9-E50E24DCCA9E"
        self.UART_TX_CHAR_UUID = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"
        self.HUB_NAME = hub_name # the name of the hub
        self.client = None
        self.nus = None
        self.rx_char = None
        self.output = None

    # functions
    def handle_disconnect(_):
        print("Hub was disconnected.")


    def handle_rx(self, _, data: bytearray):
        print("Received:", data)
        string = data.decode()
        self.output = string

    async def client_discovery(self):
        device = await BleakScanner.find_device_by_name(self.HUB_NAME)
        client = BleakClient(device, disconnected_callback=self.handle_disconnect)
        print(client)
        self.client = client

    async def connect(self):
        try:
            # Connect and get services.
            await self.client.connect()
            await self.client.start_notify(self.UART_TX_CHAR_UUID, self.handle_rx)
            self.nus = self.client.services.get_service(self.UART_SERVICE_UUID)
            self.rx_char = self.nus.get_characteristic(self.UART_RX_CHAR_UUID)
        except Exception as e:
            # Handle exceptions.
            print(e)

    # Shorthand for sending some data to the hub.
    async def send(self, data):
        await self.client.write_gatt_char(self.rx_char, data)

    async def disconnect(self):
        await self.client.disconnect()

