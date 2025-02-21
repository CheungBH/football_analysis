


class FrameQueue:
    def __init__(self, max_size=50):
        self.queue = []
        self.max_size = max_size

    def push_frame(self, frame):
        self.queue.append(frame)
        if len(self.queue) > self.max_size:
            self.queue.pop(0)

    def get_frames(self):
        return self.queue
