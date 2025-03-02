

class FlagManager:
    def __init__(self, actions, frame_duration=600, ):
        self.frame_duration = frame_duration
        self.actions = actions
        self.action_accumulate_times = {action: 0 for action in actions}
        self.current_flag = {action: False for action in actions}

    def update(self, whole, individuals):
        # self.action_times[action] += 1
        for (action, flag) in whole.items():
            if flag:
                self.action_accumulate_times[action] += 1
        for (action, flag) in individuals.items():
            flag = self.merge_individual_flags(individuals)
            if flag:
                self.action_accumulate_times[action] += 1
        self.update_counter()

    def merge_individual_flags(self, individuals):
        for action in self.actions:
            if action not in individuals[0]:
                pass
            for individual in individuals:
                if individual[action]:
                    return True
        return False

    def update_counter(self):
        for action in self.actions:
            if self.action_accumulate_times[action] > 0:
                self.action_accumulate_times[action] += 1
            if self.action_accumulate_times[action] >= self.frame_duration:
                self.action_accumulate_times[action] = 0

    def get_flag(self):
        flags = []
        for action in self.actions:
            if self.current_flag[action] and self.action_accumulate_times[action] == 0:
                flags.append(action)
        return flags
