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

