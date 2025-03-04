

action_to_int = {
    "ball_out_range": 8,
    "reverse_moving": 1,
    "low_speed_with_ball": 2,
    "not_moving_with_ball": 3,
    "lack_pressure": 4,
    "ballkeeper_change": 5,
    "delay_restart": 9,
    "poor_position": 10,
    "commit_foul": 11,
    "goalkeeper_single": 12,
}


class FlagManager:
    def __init__(self, actions, frame_duration=600, min_activate_flag=50):
        self.min_activate_flag = 50
        self.frame_duration = frame_duration
        self.actions = actions
        self.action_accumulate_times = {action: -1 for action in actions}
        # self.current_flag = {action: False for action in actions}
        self.frame_cnt = 0
        self.flag_names = []

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
                self.action_accumulate_times[action] = -1
            else:
                self.action_accumulate_times[action] += 1

    def get_flag(self):
        flags = []
        for action in self.flag_names:
            if self.action_accumulate_times[action] == 0:
                flags.append(action_to_int[action])
        return flags
