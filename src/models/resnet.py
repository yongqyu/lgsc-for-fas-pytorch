import tensorflow as tf
import tensorflow.keras as K
import tensorflow.keras.layers as layers
import tensorflow_addons as tfa

from classification_models.tfkeras import Classifiers

from typing import List


# https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
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


def conv1x1(in_planes, out_planes, stride=1):
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
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer() # planes)
        self.relu = layers.ReLU() # inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer() # planes)
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
            conv1x1(inplanes, planes * block.expansion, stride),
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
        ResNet18, preprocess_input = Classifiers.get('resnet18')
        resnet18 = ResNet18((224,224,3), weights='imagenet')
        # resnet18.fc1 = None
        # print([(i,x.name) for i,x in enumerate(resnet18.layers)]); exit()

        self.resnet18 = K.Model(inputs=resnet18.layers[0].input,
                                outputs=[resnet18.layers[5].output,
                                         resnet18.layers[26].output,
                                         resnet18.layers[45].output,
                                         resnet18.layers[64].output,
                                         resnet18.layers[83].output,
                                         ]) # (83, 'add_7'), (84, 'bn1'), (85, 'relu1')
        self.out_indices = out_indices

        self._freeze_encoder()

    def _freeze_encoder(self):
        self.resnet18.trainable = False

    def call(self, input):
        outs = self.resnet18(input)
        outs = [outs[0]] + [outs[i] for i in self.out_indices]
        # print(len(outs), [x.shape for x in outs])
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
        resnet18 = ResNet18((224,224,3), weights='imagenet')

        self.resnet18 = K.Model(inputs=resnet18.layers[0].input,
                                outputs=resnet18.get_layer('pool1').output)
        self.drop = layers.Dropout(dropout)
        self.fc = layers.Dense(
            units=num_classes
        )
        self._freeze_clf()

    def _freeze_clf(self):
        self.resnet18.trainable = False
        self.fc.trainable = True

    def call(self, input):
        x = self.resnet18(input)
        x = layers.Flatten()(x)
        x = self.drop(x)
        x = self.fc(x)
        return x
