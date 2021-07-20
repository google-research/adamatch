# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import functools

import jax.numpy as jn
import objax
from objax.typing import JaxArray
from objax.zoo.resnet_v2 import ResNet101, load_pretrained_weights_from_keras

from shared.zoo.wideresnet import WideResNet

ARCHS = 'wrn28-2 wrn28-4 wrn34-2 wrn40-2 resnet101 resnet101pretrain'.split()

class preprocess(objax.nn.Sequential):

    @staticmethod
    def _swap_channel(x):
        return x[:, ::-1, :, :]

    @staticmethod
    def _scale_back(x):
        return (x * 128) + 127.5

    def _subtract_mean(self, x):
        return x - jn.array([103.939, 116.779, 123.68])[None, :, None, None]

    def __init__(self):
        ops = [self._swap_channel, self._scale_back, self._subtract_mean]
        super().__init__(ops)


def resnet(cls, colors, nclass, bn=objax.nn.BatchNorm2D, **kwargs):
    return cls(colors, nclass, normalization_fn=bn, **objax.util.local_kwargs(kwargs, ResNet101))

def resnet_pretrained(cls, colors, nclass, bn=objax.nn.BatchNorm2D, **kwargs):
    preprocess_input = preprocess()
    model = cls(include_top=False, num_classes=nclass)
    return objax.nn.Sequential(preprocess_input + model)

def network(arch: str):
    if arch == 'wrn28-2':
        return functools.partial(WideResNet, scales=3, filters=32, repeat=4)
    elif arch == 'wrn28-4':
        return functools.partial(WideResNet, scales=3, filters=64, repeat=4)
    elif arch == 'wrn34-2':
        return functools.partial(WideResNet, scales=4, filters=32, repeat=4)
    elif arch == 'wrn40-2':
        return functools.partial(WideResNet, scales=5, filters=32, repeat=4)
    elif arch == 'resnet101':
        return functools.partial(resnet, cls=ResNet101)
    elif arch == 'resnet101pretrain':
        return functools.partial(resnet_pretrained, cls=functools.partial(load_pretrained_weights_from_keras,
            arch='ResNet101'))
    raise ValueError('Architecture not recognized', arch)
