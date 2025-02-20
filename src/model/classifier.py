from model.clip import CLIP

class Classifier(object):
    def __init__(self, config):
        self.config = config
        self.clip_model = CLIP()

    def encode(self, image, text):
        pass

    def predict(self, image, labels, prompt):
        pass