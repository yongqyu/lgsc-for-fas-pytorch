import tensorflow as tf
import tensorflow.keras as K
import tensorflow.keras.layers as layers
import tensorflow_addons as tfa

from models.resnet import make_layer, BasicBlock


class Decoder(K.Model):
    def __init__(
        self,
        in_channels=(64, 64, 128, 256, 512),
        out_channels=(512, 256, 128, 64, 64, 3),
        num_outs=6,
    ):
        super().__init__()
        assert isinstance(in_channels, tuple)
        self.in_channels = in_channels  # [64, 64, 128, 256, 512]
        self.out_channels = out_channels  # [512, 256, 128, 64, 64, 3]
        self.num_ins = len(in_channels)
        self.num_outs = num_outs

        self.deres_layers = []
        self.conv2x2 = []
        self.conv1x1 = []
        for i in range(self.num_ins - 1, -1, -1):  # 43210
            deres_layer = make_layer(
                BasicBlock,
                inplanes=128 if i == 1 else in_channels[i],
                planes=out_channels[-i - 1],
                blocks=2,
                stride=1,
                dilation=1,
                norm_layer=tfa.layers.InstanceNormalization,
            )
            out2x2 = in_channels[i] if i < 2 else int(in_channels[i] / 2)
            conv2x2 = K.Sequential([
                layers.Conv2D(filters=out2x2, kernel_size=2),
                tfa.layers.InstanceNormalization(),
                layers.ReLU(),
            ])
            conv1x1 = K.Sequential([
                layers.Conv2D(
                    filters=out_channels[-i - 1],
                    kernel_size=1,
                ),
                tfa.layers.InstanceNormalization(),
            ])
            self.deres_layers.append(deres_layer)
            self.conv2x2.append(conv2x2)
            self.conv1x1.append(conv1x1)


    def call(self, inputs):
        assert len(inputs) == len(self.in_channels)

        outs = []
        out = inputs[-1]
        outs.append(out)

        for i in range(self.num_ins):
            out = layers.UpSampling2D(size=2, interpolation="nearest")(out)
            out = tf.pad(out, tf.constant([[0, 0], [0, 1], [0, 1], [0, 0]]))
            out = self.conv2x2[i](out)
            if i < 4:
                out = tf.concat([out, inputs[-i - 2]], axis=-1)
            identity = self.conv1x1[i](out)
            out = self.deres_layers[i](out) + identity
            outs.append(out)
        outs[-1] = tf.math.tanh(outs[-1])

        return outs
