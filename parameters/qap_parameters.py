class QAPParameters:
    def __init__(self, distance: list, flow: list):
        self.distance = distance
        self.flow = flow
        self.problem_size = len(distance)
