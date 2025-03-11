

action_to_int = {
    "ball_out_range": 8,
    "reverse_moving": 1,
    "low_speed_with_ball": 2,
    "ball_change_multiple_time": 6,
    "not_moving_with_ball": 3,
    "lack_pressure": 4,
    "ballkeeper_change": 5,
    "delay_restart": 9,
    "poor_position": 10,
    "commit_foul": 11,
    "goalkeeper_single": 12,
    "outpace": 7
}


class FlagManager:
    def __init__(self, actions, frame_duration=600, min_activate_flag=30, multiple_time=4, delay_duration=10):
        self.min_activate_flag = min_activate_flag
        self.frame_duration = frame_duration
        self.multiple_time = multiple_time
        self.actions = actions
        self.delay_duration = delay_duration
        self.action_accumulate_times = {action: -delay_duration for action in actions}
        # self.current_flag = {action: False for action in actions}
        self.frame_cnt = 0
        self.flag_names = []
        self.ball_keeper_change_time = 0


    def update(self, whole, individuals):
        self.frame_cnt += 1
        self.flag_names = []
        if self.frame_cnt < self.min_activate_flag:
            return
        # self.action_times[action] += 1
        for (action, flag) in whole.items():
            if flag:
                self.flag_names.append(action)
                # self.action_accumulate_times[action] += 1
        # actions = individuals[0].keys()
        for (action, _) in individuals[0].items():
            flag = self.merge_individual_flags(individuals, action)
            if flag:
                self.flag_names.append(action)
                if action == "ballkeeper_change" and flag:
                    self.ball_keeper_change_time += 1
        self.update_counter()

    def merge_individual_flags(self, individuals, action):
        # for action in self.actions:
        # if action not in individuals[0]:
        #     pass
        for individual in individuals:
            if individual[action]:
                return True
        return False

    def update_counter(self):
        for action in self.flag_names:
            if self.action_accumulate_times[action] >= self.frame_duration:
                self.action_accumulate_times[action] = -self.delay_duration
            else:
                self.action_accumulate_times[action] += 1

        for action in self.actions:
            if action not in self.flag_names:
                if self.action_accumulate_times[action] != -self.delay_duration:
                    self.action_accumulate_times[action] += 1

    def get_flag(self):
        flags = []
        for action in self.actions:
            if self.action_accumulate_times[action] == 0:
                flags.append(action_to_int[action])
        if self.ball_keeper_change_time == self.multiple_time:
            flags.append(action_to_int["ball_change_multiple_time"])
            self.ball_keeper_change_time = 10000
        return flags
