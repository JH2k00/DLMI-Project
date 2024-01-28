from PyQt5.QtCore import QObject, pyqtSignal
import torch
from feature_extraction import calc_allfeatures
import numpy as np

class Model_Worker(QObject):

    output_signal = pyqtSignal(tuple)

    def __init__(self, models, device):
        super(Model_Worker, self).__init__()
        self.models = models
        self.device = device

    def run(self):
        pass

    def run_model(self, args):
        image, model_idx, tab_idx = args
        model = self.models[model_idx]

        if(model_idx == 0): # Deep Learning
            with torch.no_grad():
                image = image.to(self.device).unsqueeze(0)
                output = model(image).argmax(dim=1).bool().cpu().item()
        else: # Machine Learning
            image = calc_allfeatures(image.numpy())[None, :]
            output = np.argmax(model.predict(image), axis=1).astype(bool)[0]
        self.output_signal.emit((output, tab_idx, model_idx))
