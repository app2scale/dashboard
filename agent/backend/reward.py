class Reward1():
    '''Reward is simply CPU utilization, nothing more: higher cpu utilization, higher reward.

#### Formula
$$ r = c $$
where $r$ and $c$ are reward and cpu utilizations, respectively.
    '''

    def __init__(self):
        self.label = """Reward1: prefer high cpu utilization"""

    def calculate(self, state):
        return state["cpu_usage"]
    

class Reward2():
    '''Reward is the negative of CPU utilization: hence lower cpu utilization, higher reward.

#### Formula
$$ r = - c $$
where $r$ and $c$ are reward and cpu utilizations, respectively.
    '''

    def __init__(self):
        self.label = """Reward2: prefer low cpu utilization"""

    def calculate(self, state):
        return -state["cpu_usage"]