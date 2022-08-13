#! /usr/bin/env python3


speed = 100
angle_step = 2

commands = {
    "stop": "S0",
    "go": f"S{speed}",
    "left": f"I{angle_step}",
    "right": f"D{angle_step}",
}


while True:
    import serial

    i = input("intervalue")
    ArduinoSerial = serial.Serial("/dev/ttyACM0", 9600)
    ArduinoSerial.open()
    ArduinoSerial.reset_input_buffer()
    ArduinoSerial.write(commands[i].encode("utf-8"))
    ArduinoSerial.close()
