import tensorflow as tf

from models.decoder import Decoder
from models.resnet import ResNet18Classifier, ResNet18Encoder


class SCAN(tf.keras.Model):
    def __init__(self, dropout: float = 0.5):
        super().__init__()
        self.backbone = ResNet18Encoder()
        self.decoder = Decoder()
        self.clf = ResNet18Classifier(dropout=dropout)

    def call(self, x, training):
        outs = self.backbone(x, training)
        outs = self.decoder(outs)

        s = x + outs[-1]
        clf_out = self.clf(s, training)

        return outs, clf_out
