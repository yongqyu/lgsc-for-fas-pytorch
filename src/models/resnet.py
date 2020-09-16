import tensorflow as tf
import tensorflow.keras as K
import tensorflow.keras.layers as layers
import tensorflow_addons as tfa

from classification_models.tfkeras import Classifiers

from typing import List

class BatchNormalization(tf.keras.layers.BatchNormalization):
    """
    Replace BatchNormalization layers with this new layer.
    This layer has fixed momentum 0.9.
    """
    def __init__(self, momentum=0.9, name=None, **kwargs):
        super(BatchNormalization, self).__init__(momentum=0.9, name=name, **kwargs)

    def call(self, inputs, training=None):
        return super().call(inputs=inputs, training=training)

    def get_config(self):
        config = super(BatchNormalization, self).get_config()
        return config
tf.keras.layers.BatchNormalization = BatchNormalization

# https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py


def conv3x3(out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return layers.Conv2D(
        out_planes,
        kernel_size=3,
        strides=stride,
        padding='same',
        groups=groups,
        use_bias=False,
        dilation_rate=dilation,
    )


def conv1x1(out_planes, stride=1):
    """1x1 convolution"""
    return layers.Conv2D(
        out_planes, kernel_size=1, strides=stride, use_bias=False
    )


class BasicBlock(K.Model):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = layers.BatchNormalization
        if groups != 1 or base_width != 64:
            raise ValueError(
                "BasicBlock only supports groups=1 and base_width=64"
            )
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock"
            )
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(planes, stride)
        self.bn1 = norm_layer()
        self.relu = layers.ReLU()
        self.conv2 = conv3x3(planes)
        self.bn2 = norm_layer()
        self.downsample = downsample
        self.stride = stride

    def call(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


def make_layer(
    block, inplanes, planes, blocks, stride=1, dilation=1, norm_layer=None
):
    if norm_layer is None:
        norm_layer = layers.BatchNormalization

    if stride != 1 or inplanes != planes * block.expansion:
        downsample = K.Sequential([
            conv1x1(planes * block.expansion, stride),
            norm_layer(), # planes * block.expansion),
        ])

    layers = []
    layers.append(
        block(
            inplanes,
            planes,
            stride=stride,
            downsample=downsample,
            groups=1,
            base_width=64,
            dilation=dilation,
            norm_layer=norm_layer,
        )
    )
    inplanes = planes * block.expansion

    for _ in range(1, blocks):
        layers.append(
            block(
                inplanes,
                planes,
                groups=1,
                base_width=64,
                dilation=dilation,
                norm_layer=norm_layer,
            )
        )

    return K.Sequential([*layers])


class ResNet18Encoder(K.Model):
    def __init__(
        self, out_indices: List[int] = (1, 2, 3, 4), pretrained: bool = True
    ):
        super().__init__()
        resnet18 = K.models.load_model('/data/project/rw/ASDG/tf_ssdg/tf_resnet18')

        self.resnet18 = K.models.Model(inputs=resnet18.input,
                                       outputs=[resnet18.get_layer('relu').output,
                                                resnet18.get_layer('activation_3').output,
                                                resnet18.get_layer('activation_7').output,
                                                resnet18.get_layer('activation_11').output,
                                                resnet18.get_layer('activation_15').output
                                                ])

        self.out_indices = out_indices

        self._freeze_encoder()

    def _freeze_encoder(self):
        self.resnet18.trainable = False

    def unfreeze_encoder(self):
        self.resnet18.trainable = True

    def call(self, input, training):
        outs = self.resnet18(input, training)
        outs = [outs[0]] + [outs[i] for i in self.out_indices]
        return outs


class ResNet18Classifier(K.Model):
    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.5,
    ):
        super().__init__()
        ResNet18, preprocess_input = Classifiers.get('resnet18')
        resnet18 = K.models.load_model('/data/project/rw/ASDG/tf_ssdg/tf_resnet18')

        self.resnet18 = K.models.Model(inputs=resnet18.input,
                                       outputs=resnet18.get_layer('avgpool').output)
        self.drop = layers.Dropout(dropout)
        self.fc = layers.Dense(
            units=num_classes,
            # activation='softmax'
        )
        self._freeze_clf()

    def _freeze_clf(self):
        self.resnet18.trainable = False
        self.fc.trainable = True

    def unfreeze_clf(self):
        self.resnet18.trainable = True

    def call(self, input, training):
        x = self.resnet18(input, training)
        x = layers.Flatten()(x)
        if training:
            x = self.drop(x)
        x = self.fc(x)
        return x
