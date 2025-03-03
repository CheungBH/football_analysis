from collections import defaultdict
from .criterion import *

# low_speed -- ball_out_range -- goalkeeper_single -- reverse
checkers = {
    "low_speed": LowSpeedChecker,
    "reverse_moving": MovingReverseChecker,
    "delay_restart": DelayRestartChecker,
    "not_moving_with_ball": StandingChecker,
    "low_speed_with_ball": LowSpeedWithBallChecker,
    #"side_referee": SideRefereeChecker,
    "goalkeeper_single": GoalKeeperSingleChecker,
    "ball_out_range": BallOutRangeChecker,
    "commit_foul": CommitFoulChecker,
    "lack_pressure": LackPressureChecker,
    "poor_position": PoorPositionChecker,
    "ballkeeper_change": BallKeeperChangeChecker,
}

class AnalysisManager:

    def __init__(self, check_list, court, display_x=100):
        # self.display_x = display_x
        self.criterion = [checkers[check_item](court=court, display_x=display_x) for check_item in check_list]
        self.flag = 0
        self.team_dict = defaultdict(list)
        self.ball_exit = None
        self.flag_dict = {}

    def process(self, players, balls,frame_id,matrix,frame_queue):
        self.flag_dict = {}
        for p_id, location in players.items():
            self.team_dict[p_id].append(location)
        for key in self.team_dict:
            if key not in players:
                self.team_dict[key].append([-1, -1])
        for criterion in self.criterion:
            criterion.process(players=self.team_dict,player_current = players, balls=balls,frame_queue = frame_queue,frame_id=frame_id, matrix=matrix)


    def visualize(self, frame):
        for c_idx, criterion in enumerate(self.criterion):
            criterion.visualize_details(frame, c_idx)
            self.flag_dict[criterion.name] = criterion.flag
            # self.flag_list.append([criterion.name,criterion.flag])
        for idx,flag in enumerate(self.flag_dict):
            if self.flag_dict[flag] == 1:
                print(flag +' is activated')
        self.flag= sum(self.criterion[i].flag for i in range(len(self.criterion)))
