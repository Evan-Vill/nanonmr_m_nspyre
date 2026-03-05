import serial
import sys
import glob

# import serial.tools.list_ports

# ports = serial.tools.list_ports.comports()

# for port in ports:
#     print(f"{port.device}: {port.description}")

# import serial.tools.list_ports

# ports = serial.tools.list_ports.comports()
# for port in ports:
#     print(f"Port: {port.device}")
#     print(f"  Description: {port.description}")
#     print(f"  HWID: {port.hwid}")
#     print(f"  VID: {port.vid}")
#     print(f"  PID: {port.pid}")
#     print(f"  Serial number: {port.serial_number}")
#     print("")

# import time
# for baud in (115200, 57600, 38400, 19200, 9600):
#     try:
#         with serial.Serial('COM4', baudrate=baud, timeout=0.5) as ser:
#             ser.reset_input_buffer(); ser.reset_output_buffer()
#             ser.write(b'?HID\r'); ser.flush()
#             time.sleep(0.05)
#             rep = ser.read_until(b'\r') or ser.read_until(b'\n')
#             print(baud, rep)
#     except Exception as e:
#         print(baud, 'ERR', e)

from serial.tools import list_ports
for p in list_ports.comports():
    print(p.device, p.description, p.manufacturer, hex(p.vid) if p.vid else None, hex(p.pid) if p.pid else None)


TIMEOUT = 0.5
TIMEOUT_INIT = 0.1


class LasersList():
    def __init__(self):
        self.count = 0
        self.laserslist = []
        # Pin the laser’s port; do NOT scan all ports (prevents touching Zaber)
        self.ports_com = self.serial_ports()
        # self.ports_com = ['COM4']
        print("Detected COM ports:", self.ports_com)

        # Try common Oxxius baudrates (115200 often used after firmware/service)
        candidate_bauds = [115200, 57600, 38400, 19200, 9600]
        ok = False

        for com in self.ports_com:
            found = False
            for baud in candidate_bauds:
                ser = serial.Serial()
                ser.port = com
                ser.baudrate = baud
                ser.timeout = TIMEOUT_INIT

                try:
                    hid = self.send(ser, "?HID", True)
                except Exception as e:
                    # Non-ASCII or timeout → try next baud
                    continue

                if hid and hid.startswith("LAS"):
                    try:
                        inf = self.send(ser, "inf?", False)
                        infos = inf.split('-')
                        type_ = infos[0] if len(infos) > 0 else ""
                        couleur = infos[1] if len(infos) > 1 else ""
                        puissance = infos[2] if len(infos) > 2 else ""
                        serial_number = hid.split(',')[0]
                        self.laserslist.append([serial_number, type_, couleur, puissance, com, baud])
                        found = True
                        break
                    except Exception:
                        # HID ok but inf? failed → still a baud issue; try next
                        continue

            if not ok:
                raise RuntimeError(
                    f"No valid Oxxius reply on {self.ports_com} at {candidate_bauds}. "
                    "Check port, baudrate, cable, and line ending (CR)."
                )

            print(f"[OK] Oxxius {self.serial_number} on {self.port} @ {self.baud}")


    def send(self, ser, command, init):
        import time
        try:
            if not ser.is_open:
                ser.open()

            # Make sure the serial params are explicit & sane
            ser.bytesize = serial.EIGHTBITS
            ser.parity   = serial.PARITY_NONE
            ser.stopbits = serial.STOPBITS_ONE
            ser.write_timeout = 0.5

            # Clean any junk before we talk
            ser.reset_input_buffer()
            ser.reset_output_buffer()

            if init:
                # Some firmwares like a wake-up; CR (not CRLF) is safest
                ser.write(b"\r")
                ser.flush()
                time.sleep(0.03)
                _ = ser.read(ser.in_waiting or 1)  # drain anything echoed

            # Send the command with CR (Oxxius typically expects '\r')
            ser.write((command + "\r").encode("ascii"))
            ser.flush()

            # Read one line ending in either CR or LF (don’t rely on '\n' only)
            rep_bytes = ser.read_until(expected=b"\r")
            if rep_bytes.endswith(b"\r"):
                pass
            else:
                # fall back to LF if device uses CRLF or LF
                rep_bytes += ser.read_until(expected=b"\n")

            # Validate ASCII instead of blind UTF-8
            if not rep_bytes:
                return ""  # timeout → empty string like your original logic
            if not all((32 <= b <= 126) or (b in (9, 10, 13)) for b in rep_bytes):
                # Non-text indicates wrong baud/port or binary banner
                raise RuntimeError(f"Non-ASCII response from {ser.port}: {rep_bytes[:8]!r}")

            return rep_bytes.decode("ascii").replace("\r", "").replace("\n", "").strip()

        finally:
            try:
                ser.close()
            except Exception:
                pass

    
    def serial_ports(self):
        """ Lists serial port names

            :raises EnvironmentError:
                On unsupported or unknown platforms
            :returns:
                A list of the serial ports available on the system
        """
        if sys.platform.startswith('win'):
            ports = ['COM%s' % (i + 1) for i in range(256)]
        elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
            # this excludes your current terminal "/dev/tty"
            ports = glob.glob('/dev/tty[A-Za-z]*')
        elif sys.platform.startswith('darwin'):
            ports = glob.glob('/dev/tty.*')
        else:
            raise EnvironmentError('Unsupported platform')

        result = []
        for port in ports:
            try:
                s = serial.Serial(port)
                s.close()
                result.append(port)
            except (OSError, serial.SerialException):
                pass
        return result
    
    def get_list(self):
        return self.laserslist
        
    def get_serial_numbers(self):
        list = []
        for laser in self.laserslist:
            list.append(laser[0])
        return list
    def get_types(self):
        list = []
        for laser in self.laserslist:
            list.append(laser[1])
        return list
    def get_colors(self):
        list = []
        for laser in self.laserslist:
            list.append(laser[2])
        return list
    def get_powers(self):
        list = []
        for laser in self.laserslist:
            list.append(laser[3])
        return list  
    def get_coms(self):
        list = []
        for laser in self.laserslist:
            list.append(laser[4])
        return list  
    def find_serial_number(self, serial_number):
        for i,sn in enumerate(self.get_serial_numbers()):
            if sn == serial_number:
                return self.laserslist[i]
        return []
    def find_color(self, color):
        for i,co in enumerate(self.get_colors()):
            if co == color:
                return self.laserslist[i]
        return []
    def find_com(self, com):
        for i,co in enumerate(self.get_coms()):
            if co == com:
                return self.laserslist[i]
        return []       


class Laser():
    def __init__(self, laser_infos):
        self.serial_number = laser_infos[0]
        self.type = laser_infos[1]
        self.color = laser_infos[2]
        self.power = laser_infos[3]
        
        self.ser = serial.Serial()
        self.ser.port = laser_infos[4]
        self.ser.baudrate = laser_infos[5]
        self.ser.timeout = TIMEOUT
        
    def open(self):
        self.ser.open()
        self.ser.write(b"dummy\r\n")
        rep = self.ser.readline()
        
    def close(self):
        self.ser.close()
        
    def send(self, command):
        self.ser.write((command+"\r\n").encode('ASCII'))
        rep = self.ser.readline().decode().replace("\r", "").replace("\n", "")
        return rep

