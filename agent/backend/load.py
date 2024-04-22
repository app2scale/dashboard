import numpy as np

class ConstantLoad():
    '''Constant load profile'''

    def __init__(self, load):
        self.label = f"""Constant load: {load}"""
        self.load = load

    def __iter__(self):
        self.step = 0
        return self
    
    def __next__(self):
        self.step += 1
        eod = False
        return self.load, eod

class SinusLoad():
    '''Periodic load profile'''

    def __init__(self, amplitude, period):
        self.label = f"""Sinusodial load"""
        self.amplitude = amplitude
        self.period = period

    def __iter__(self):
        self.step = 0
        return self
    
    def __next__(self):
        self.step += 1
        eod = True if self.step % self.period == 0 else False
        return max(0, self.amplitude + self.amplitude * np.sin(2 * np.pi * (self.step % self.period) / self.period )), eod


class PaymentGateway113Load():
    '''Payment Gateway Load 113 Days

Incoming load to Payment Gateway.

    '''

    def __init__(self):
        self.label = f"""Payment Gateway 113 days"""
        self.loadprofile = [40, 32, 48, 48, 56, 48, 40, 32, 64, 64, 64, 64, 72, 64, 48, 32, 80, 64, 64, 56, 56,
                     48, 40, 64, 56, 56, 56, 56, 48, 48, 96, 64, 64, 64, 72, 80, 104, 96, 112, 136, 80,
                     72, 88, 96, 104, 144, 136, 176, 112, 128, 104, 112, 88, 88, 112, 104, 104, 120, 128,
                     88, 88, 160, 144, 152, 160, 208, 128, 96, 128, 112, 112, 144, 136, 136, 144, 144, 128,
                     152, 248, 216, 232, 216, 152, 152, 120, 152, 144, 144, 136, 144, 120, 120, 152, 184, 216, 2,
                     32, 240, 152, 120, 168, 144, 136, 136, 144, 112, 112, 152, 152, 152, 152, 144, 120, 112]
        self.n = len(self.loadprofile)

    def __iter__(self):
        self.step = 0
        return self

    def __next__(self):
        i = self.step % self.n
        eod = True if i == self.n - 1 else False
        self.step += 1
        return self.loadprofile[i], eod