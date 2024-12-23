
from .criterion import *

class AnalysisManager:

    def __init__(self):
        self.criterion = [LowSpeedChecker()]

    def process(self, team1, team2):
        status = []
        for criterion in self.criterion:
            status += criterion.process([team1, team2])
        # self.low_speed_checker.process([team1])
        return status

    def visualize(self, frame):

        for criterion in self.criterion:
            criterion.visualize(frame)