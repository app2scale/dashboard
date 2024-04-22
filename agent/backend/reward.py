class RewardHighCPUUsage():
    '''Promotes high CPU usage.

Reward is simply equivalent to cpu usage. When there is more than one pod running,
the cpu usage is the average cpu usage over pods.

#### Formula
$$ r = c $$
where $r$ and $c$ are reward and cpu usage, respectively.
    '''

    def __init__(self):
        self.label = """RewardHighCPUUsage"""

    def calculate(self, state):
        return state["cpu_usage"]
    
class RewardLowCPUUsage():
    '''Promotes low CPU usage.

Reward is simply equivalent to the negative of cpu usage. When there is more than one pod running,
the cpu usage is the average cpu usage over pods.

#### Formula
$$ r = - c $$
where $r$ and $c$ are reward and cpu usage, respectively.
    '''

    def __init__(self):
        self.label = """RewardLowCPUUsage"""

    def calculate(self, state):
        return -state["cpu_usage"]
    