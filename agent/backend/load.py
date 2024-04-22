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
        return self.load

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
        return max(0, self.amplitude + self.amplitude * np.sin(2 * np.pi * (self.step % self.period) / self.period ))