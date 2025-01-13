
from .criterion import *

checkers = {"low_speed": LowSpeedChecker,
            "reverse_moving": MovingReverseChecker,
            "side_referee": SideRefereeChecker,
            "ballkeeper_change": BallKeeperChangeChecker,
            "goalkeeper_single": GoalKeeperSingleChecker,
            "ball_out_range": BallOutRangeChecker,
            }

class AnalysisManager:

    def __init__(self, check_list, court):
        self.criterion = [checkers[check_item](court) for check_item in check_list]

    def process(self, team1_players, team2_players, balls, side_referees, goalkeepers):
        for criterion in self.criterion:
            criterion.process(players1=team1_players, players2=team2_players, balls=balls,
                              side_referees=side_referees, goalkeepers=goalkeepers)

    def visualize(self, frame):
        for criterion in self.criterion:
            criterion.visualize(frame)
