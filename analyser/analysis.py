from collections import defaultdict
from .criterion import *
# low_speed -- ball_out_range -- goalkeeper_single -- reverse
checkers = {"low_speed": LowSpeedChecker,
            "reverse_moving": MovingReverseChecker,
            #"side_referee": SideRefereeChecker,
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
        self.team_dict = defaultdict(list)

        self.flag_list=[]

    # def process(self, team1_players, team2_players, balls, side_referees, goalkeepers1,goalkeepers2, frame_id,matrix,frame_queue):
    #     for criterion in self.criterion:
    #         if goalkeepers1:
    #             criterion.process(players1=team1_players, players2=team2_players, balls=balls,frame_queue = frame_queue,
    #                           side_referees=side_referees, goalkeepers=goalkeepers1, frame_id=frame_id, matrix=matrix)
    #         else:
    #             criterion.process(players1=team1_players, players2=team2_players, balls=balls,frame_queue = frame_queue,
    #                           side_referees=side_referees, goalkeepers=goalkeepers2, frame_id=frame_id, matrix=matrix)
    #         self.flag= sum(self.criterion[i].flag for i in range(len(self.criterion)))
    def process(self, players, balls,frame_id,matrix,frame_queue):
        self.flag_list=[]
        for p_id, location in players.items():
            self.team_dict[p_id].append(location)
        for key in self.team_dict:
            if key not in players:
                self.team_dict[key].append([-1, -1])
        for criterion in self.criterion:
            criterion.process(players=self.team_dict, balls=balls,frame_queue = frame_queue,frame_id=frame_id, matrix=matrix)


    def visualize(self, frame):
        for criterion in self.criterion:
            criterion.visualize_details(frame)
            self.flag_list.append([criterion.name,criterion.flag])
        for idx,flag in enumerate(self.flag_list):
            if flag[1] == 1:
                print(flag[0] +' is activated')
        self.flag= sum(self.criterion[i].flag for i in range(len(self.criterion)))
