
from .criterion import *

checkers = {"low_speed": LowSpeedChecker,
            "reverse_moving": MovingReverseChecker,
            "side_referee": SideRefereeChecker,
            "ballkeeper_change": BallKeeperChangeChecker,
            "goalkeeper_single": GoalKeeperSingleChecker,
            "ball_out_range": BallOutRangeChecker,
            #"standing": StandingChecker,
            "lack_pressure": LackPressureChecker,
            }

class AnalysisManager:

    def __init__(self, check_list, court):
        self.criterion = [checkers[check_item](court=court) for check_item in check_list]
        self.flag = 0

    def process(self, team1_players, team2_players, balls, side_referees, goalkeepers1,goalkeepers2, frame_id):
        for criterion in self.criterion:
            if goalkeepers1:
                criterion.process(players1=team1_players, players2=team2_players, balls=balls,
                              side_referees=side_referees, goalkeepers=goalkeepers1, frame_id=frame_id)
            else:
                criterion.process(players1=team1_players, players2=team2_players, balls=balls,
                              side_referees=side_referees, goalkeepers=goalkeepers2, frame_id=frame_id)
            self.flag= sum(self.criterion[i].flag for i in range(len(self.criterion)))





    def visualize(self, frame):
        for criterion in self.criterion:
            criterion.visualize_details(frame)
