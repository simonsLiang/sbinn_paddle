import numpy as np

class Real:
    def __init__(self, precision):
        self.precision = None
        self.reals = None
        if precision == 32:
            self.set_float32()
        elif precision == 64:
            self.set_float64()

    def __call__(self, package):
        return self.reals[package]

    def set_float32(self):
        self.precision = 32
        self.reals = {np: np.float32}

    def set_float64(self):
        self.precision = 64
        self.reals = {np: np.float64}
