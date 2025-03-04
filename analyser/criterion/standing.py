from .speed import SpeedChecker

class StandingChecker(SpeedChecker):
    def __init__(self, **kwargs):
        super(StandingChecker, self).__init__()
        self.name = "not_moving_with_ball"
        self.speed_threshold = 0.5
        self.green_word = "Normal speed"
        self.red_word = "Not moving"
        self.detail_word = "not moving"

