class Target():
    def __init__(self):
        pass

    def get_score(self, input):
        raise NotImplementedError("Needs to have a score function")


class SSDObjectDetectorTarget(Target):
    def __init__(self):
        super().__init__()
        self.detector = None

    def calc_loss(self, result, target):
        pass

    def get_score(self, input):
        pass


class ImageClasifier(Target):

    def get_score(self, input):
        pass
