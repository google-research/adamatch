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
import itertools
import os
from typing import List, Callable, Dict

import numpy as np

from shared.data import core
from shared.data.core import DataSet, record_parse, record_parse_mnist


class DataSetsFSL:
    def __init__(self, name: str, train: DataSet, test: Dict[str, DataSet], valid: DataSet, nclass: int = 10):
        self.name = name
        self.train = train
        self.test = test
        self.valid = valid
        self.nclass = nclass

    @property
    def image_shape(self):
        return self.train.image_shape

    @property
    def colors(self):
        return self.image_shape[2]

    @property
    def height(self):
        return self.image_shape[0]

    @property
    def width(self):
        return self.image_shape[1]

    @classmethod
    def creator(cls, name: str, train_files: List[str], test_files: Dict[str, List[str]], valid: int,
                parse_fn: Callable = record_parse,
                nclass: int = 10, height: int = 32, width: int = 32, colors: int = 3, cache: bool = False):
        train_files = [os.path.join(core.DATA_DIR, x) for x in train_files]
        test_files = {key: [os.path.join(core.DATA_DIR, x) for x in value] for key, value in test_files.items()}

        def create():
            image_shape = height, width, colors
            kw = dict(parse_fn=parse_fn)
            datasets = dict(train=DataSet.from_files(train_files, image_shape, cache=cache, **kw).skip(valid),
                            valid=DataSet.from_files(train_files, image_shape, cache=cache, **kw).take(valid),
                            test={key: DataSet.from_files(value, image_shape, cache=cache, **kw)
                                  for key, value in test_files.items()})
            return cls(name + '-' + str(valid), nclass=nclass, **datasets)

        return name + '-' + str(valid), create


def create_datasets(samples_per_class=(1, 2, 3, 4, 5, 10, 25, 100, 400)):
    samples_per_class = np.array(samples_per_class, np.uint32)
    d = {}
    d.update([DataSetsFSL.creator('mnist', ['mnist-train.tfrecord'], {'mnist': ['mnist-test.tfrecord']}, valid,
                                  cache=True, parse_fn=record_parse_mnist) for valid in [0, 5000]])
    d.update([DataSetsFSL.creator('cifar10', ['cifar10-train.tfrecord'], {'cifar10': ['cifar10-test.tfrecord']}, valid,
                                  cache=True) for valid in [0, 5000]])
    d.update(
        [DataSetsFSL.creator('cifar100', ['cifar100-train.tfrecord'], {'cifar100': ['cifar100-test.tfrecord']}, valid,
                             nclass=100, cache=True) for valid in [0, 5000]])
    d.update([DataSetsFSL.creator('svhn', ['svhn-train.tfrecord'], {'svhn': ['svhn-test.tfrecord']},
                                  valid) for valid in [0, 5000]])
    d.update([DataSetsFSL.creator('svhnx', ['svhn-train.tfrecord', 'svhn-extra.tfrecord'],
                                  {'svhn': ['svhn-test.tfrecord']}, valid) for valid in [0, 5000]])
    d.update([DataSetsFSL.creator('cifar10_1', ['cifar10-train.tfrecord'],
                                  {'cifar10': ['cifar10-test.tfrecord'], 'cifar10.1': ['cifar10.1-test.tfrecord']},
                                  valid, cache=True) for valid in [0, 5000]])
    d.update([DataSetsFSL.creator('cifar10.%d@%d' % (seed, sz), ['SSL/cifar10.%d@%d-label.tfrecord' % (seed, sz)],
                                  {'cifar10': ['cifar10-test.tfrecord']}, valid, cache=True)
              for valid, seed, sz in itertools.product([0, 5000], range(6), 10 * samples_per_class)])
    d.update([DataSetsFSL.creator('cifar100.%d@%d' % (seed, sz), ['SSL/cifar100.%d@%d-label.tfrecord' % (seed, sz)],
                                  {'cifar100': ['cifar100-test.tfrecord']}, valid, nclass=100)
              for valid, seed, sz in itertools.product([0, 5000], range(6), 100 * samples_per_class)])
    d.update([DataSetsFSL.creator('svhn.%d@%d' % (seed, sz), ['SSL/svhn.%d@%d-label.tfrecord' % (seed, sz)],
                                  {'svhn': ['svhn-test.tfrecord']}, valid)
              for valid, seed, sz in itertools.product([0, 5000], range(6), 10 * samples_per_class)])
    d.update([DataSetsFSL.creator('svhnx.%d@%d' % (seed, sz), ['SSL/svhnx.%d@%d-label.tfrecord' % (seed, sz)],
                                  {'svhn': ['svhn-test.tfrecord']}, valid)
              for valid, seed, sz in itertools.product([0, 5000], range(6), 10 * samples_per_class)])
    d.update([DataSetsFSL.creator('stl10.%d@%d' % (seed, sz), ['SSL/stl10.%d@%d-label.tfrecord' % (seed, sz)],
                                  {'stl10': ['stl10-test.tfrecord']}, valid, height=96, width=96)
              for valid, seed, sz in itertools.product([0, 5000], range(6), 10 * samples_per_class)])
    # DomainNet datasets
    categories = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
    for category, size in itertools.product(categories, [32, 64, 128, 224]):
        d.update([DataSetsFSL.creator(f'domainnet{size}_{category}',
                                      [f'domainnet{size}_{category}-train.tfrecord'],
                                      {category: [f'domainnet{size}_{category}-test.tfrecord']},
                                      valid, height=size, width=size, nclass=345,
                                      cache=size <= 64) for valid in [0, 5000]])
        d.update([DataSetsFSL.creator(
            f'domainnet{size}_no_{category}',
            [f'domainnet{size}_{t}-train.tfrecord' for t in categories if t != category],
            {f'no_{category}': [f'domainnet{size}_{t}-test.tfrecord' for t in categories if t != category]},
            valid, height=size, width=size, nclass=345, cache=size <= 64) for valid in [0, 5000]])
    # Office31 datasets
    categories = ['webcam', 'dslr', 'amazon']
    for category, size in itertools.product(categories, [32, 64, 128, 224]):
        d.update([DataSetsFSL.creator(f'office31{size}_{category}',
                                [f'office31{size}_{category}-train.tfrecord'],
                                {category: [f'office31{size}_{category}-test.tfrecord']},
                                valid, height=size, width=size, nclass=31,
                                cache=size <= 64) for valid in [0, 5000]])
    # DigitFive datasets
    categories = ['usps', 'mnist', 'mnistm', 'svhn', 'syndigit']
    for category, size in itertools.product(categories, [32]):
        d.update([DataSetsFSL.creator(f'digitfive{size}_{category}',
                                [f'digitfive{size}_{category}-train.tfrecord'],
                                {category: [f'digitfive{size}_{category}-test.tfrecord']},
                                valid, height=size, width=size, nclass=10,
                                cache=size <= 64) for valid in [0, 5000]])


    return d


DATASETS = create_datasets
