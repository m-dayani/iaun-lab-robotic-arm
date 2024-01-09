import serial
import time


class MyArduino:

    def __init__(self, port, bd, timeout=0.1):
        self.port = port
        self.bd = bd
        self.timeout = timeout
        self.arduino = None
        self.init()

    def init(self):
        try:
            self.arduino = serial.Serial(port=self.port, baudrate=self.bd, timeout=self.timeout)
            time.sleep(2)
        except serial.SerialException:
            print("Error: cannot open Arduino")

    def close(self):
        if self.arduino is not None:
            self.arduino.close()

    def transmit(self, msg):
        if self.arduino is not None:
            self.arduino.write(bytes(msg, 'utf-8'))
            time.sleep(0.05)
            msg_rx = self.arduino.readline()
            # print(msg)
            return msg_rx
        return ''

    def send(self, msg):
        if self.arduino is not None:
            self.arduino.write(bytes(msg, 'utf-8'))


if __name__ == "__main__":

    arduino = MyArduino('/dev/ttyACM0', 9600)

