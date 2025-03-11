from collections import defaultdict
from .criterion import *

# low_speed -- ball_out_range -- goalkeeper_single -- reverse
checkers = {
    "low_speed": LowSpeedChecker,
    "reverse_moving": MovingReverseChecker,
    "delay_restart": DelayRestartChecker,
    "not_moving_with_ball": StandingChecker,
    "low_speed_with_ball": LowSpeedWithBallChecker,
    "outpace": OutpaceChecker,
    #"side_referee": SideRefereeChecker,
    "goalkeeper_single": GoalKeeperSingleChecker,
    "ball_out_range": BallOutRangeChecker,
    "commit_foul": CommitFoulChecker,
    "lack_pressure": LackPressureChecker,
    "poor_position": PoorPositionChecker,
    "ballkeeper_change": BallKeeperChangeChecker,
}

class AnalysisManager:

    def __init__(self, check_list, court, display_x=100, video_path=""):
        # self.display_x = display_x
        self.criterion = [checkers[check_item](court=court, display_x=display_x, video_path=video_path) for check_item in check_list]
        self.flag = 0
        self.team_dict = defaultdict(list)
        self.ball_exit = None
        self.flag_dict = {}
        self.player_color_dict = defaultdict(list)
        self.player_img_box = defaultdict(list)

    def process(self, players, player_img_box,balls,frame_id, matrix,frame_queue,colors):
        self.flag_dict = {}
        for idx, (p_id, location) in enumerate(players.items()):
            self.team_dict[p_id].append(location)
            self.player_color_dict[p_id].append(colors[p_id])
            self.player_img_box[p_id].append(player_img_box[p_id])
        for key in self.team_dict:
            if key not in players:
                self.team_dict[key].append([-1, -1])
                self.player_img_box[key].append([-1, -1, -1, -1])
                self.player_color_dict[key].append(-1)
        for criterion in self.criterion:
            criterion.process(players=self.team_dict,player_current = players, balls=balls,frame_queue = frame_queue,
                              frame_id=frame_id, color = self.player_color_dict)


    def visualize(self, frame):
        for c_idx, criterion in enumerate(self.criterion):
            criterion.visualize_details(frame, c_idx)
            self.flag_dict[criterion.name] = criterion.flag
            # self.flag_list.append([criterion.name,criterion.flag])
        criterion_dict = {type(checker).__name__: checker for checker in self.criterion}
        for idx,flag in enumerate(self.flag_dict):
            if self.flag_dict[flag] == 1:
                print(flag +' is activated')
                if 'low_speed_with_ball' in self.flag_dict:
                    speed_checker = criterion_dict.get('SpeedChecker')
                    if speed_checker.average_speed:
                        print('Average speed is ' + str(speed_checker.average_speed[-1]))
        self.flag= sum(self.criterion[i].flag for i in range(len(self.criterion)))
