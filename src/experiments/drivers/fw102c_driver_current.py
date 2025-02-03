from pylablib.devices import Thorlabs

class FilterWheel():
    def __init__(self):
        self.wheel = Thorlabs.FWv1("COM4")

    def get_pos(self):
        self.wheel.get_position()    

    def set_pos(self, pos):
        self.wheel.set_position(pos)

    def close(self):
        self.wheel.close()