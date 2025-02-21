class HRLManager:
    def __init__(self, alert_system):
        self.alert_system = alert_system

class HRLWorker:
    def __init__(self, manager):
        self.manager = manager 

class NeuralHRL:
    def __init__(self, alert_system):
        self.alert_system = alert_system 