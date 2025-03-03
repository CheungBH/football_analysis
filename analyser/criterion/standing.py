from .speed import SpeedChecker


class StandingChecker(SpeedChecker):
    def __init__(self, **kwargs):
        super(StandingChecker, self).__init__()
        self.name = "standing"
        self.speed_threshold = 100
        self.green_word = "Normal speed"
        self.red_word = "Not moving"
        self.detail_word = "not moving"

