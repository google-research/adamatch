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
import time
from typing import List, Callable, Tuple

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from shared.data import core
from shared.data.core import record_parse_mnist, DataSet, record_parse, label_parse


class DataSetSSL:
    def __init__(self, name: str, train: DataSet, nclass: int = 10):
        self.name = name
        self.train = train
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
    def creator(cls, name: str, train_files: List[str], parse_fn: Callable = record_parse,
                nclass: int = 10, height: int = 32, width: int = 32, colors: int = 3, cache: bool = False):
        train_files = [os.path.join(core.DATA_DIR, x) for x in train_files]

        def create(samples_per_class: int, seed: int):
            target_file = os.path.join(core.DATA_DIR, f'{name}({samples_per_class},seed={seed}).tfrecord')
            if not os.path.exists(target_file):
                cls.materialize_subset(target_file, train_files, samples_per_class, seed, nclass)
            image_shape = height, width, colors
            train = DataSet.from_files([target_file], image_shape, cache=cache, parse_fn=parse_fn)
            return cls(name, nclass=nclass, train=train)

        return name, create

    @staticmethod
    def parse_name(name: str) -> Tuple[str, int, int]:
        try:
            name, params = name.split('(')
            params = params.split(',')
            samples_per_class = int(params[0])
            seed = int(params[1][5:-1])
        except:
            raise ValueError(f'Name "{name}" must be of the form name(int,seed=int).')
        return name, samples_per_class, seed

    @staticmethod
    def materialize_subset(target_file: str, train_files: List[str], samples_per_class: int, seed: int, nclass: int):
        print(f'Materializing subset {target_file}')
        print(f'\015    {"Samples per class":32s}', samples_per_class)
        print(f'\015    {"Random seed":32s}', seed)
        t0 = time.time()
        train = DataSet.from_files(train_files, (0, 0, 0), parse_fn=label_parse)
        class_to_idx = [[] for _ in range(nclass)]
        for batch in tqdm(train.parse().batch(1024), leave=False, desc='Building class map'):
            for idx, label in zip(batch['index']._numpy(), batch['label']._numpy()):
                class_to_idx[label].append(idx)
        print(f'\015    {"Number of source samples":32s}', sum(len(x) for x in class_to_idx))
        np.random.seed(seed)
        class_to_idx = [np.random.choice(x, samples_per_class, replace=True) for x in class_to_idx]
        keep_idx = set()
        for x in class_to_idx:
            keep_idx |= set(x)
        print(f'\015    {"Number of target samples":32s}', sum(len(x) for x in class_to_idx))
        with tf.io.TFRecordWriter(target_file + '.tmp') as writer:
            for index, record in tqdm(train, leave=False, desc=f'Saving dataset f{target_file}'):
                if index._numpy() not in keep_idx:
                    continue
                writer.write(record._numpy())
        os.rename(target_file + '.tmp', target_file)
        print(f'\015    {"File size":32s}', os.path.getsize(target_file))
        print(f'\015    Completed in {int(time.time() - t0)}s')


def create_datasets():
    d = {}
    d.update([DataSetSSL.creator('cifar10', ['cifar10-train.tfrecord'], cache=True)])
    d.update([DataSetSSL.creator('mnist', ['mnist-train.tfrecord'], cache=True, parse_fn=record_parse_mnist)])
    d.update([DataSetSSL.creator('svhn', ['svhn-train.tfrecord'])])
    d.update([DataSetSSL.creator('svhnx', ['svhn-train.tfrecord', 'svhn-extra.tfrecord'])])
    # DomainNet datasets
    categories = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
    for category, size in itertools.product(categories, [32, 64, 128, 224]):
        d.update([DataSetSSL.creator(f'domainnet{size}_{category}', [f'domainnet{size}_{category}-train.tfrecord'],
                                     height=size, width=size, nclass=345, cache=size <= 64)])
    # Office31 datasets
    categories = ['webcam', 'dslr', 'amazon']
    for category, size in itertools.product(categories, [32, 64, 128, 224]):
        d.update([DataSetSSL.creator(f'office31{size}_{category}', [f'office31{size}_{category}-train.tfrecord'],
                                     height=size, width=size, nclass=31, cache=size <= 64)])
    # DigitFive datasets
    categories = ['usps', 'mnist', 'mnistm', 'svhn', 'syndigit']
    for category, size in itertools.product(categories, [32]):
        d.update([DataSetSSL.creator(f'digitfive{size}_{category}', [f'digitfive{size}_{category}-train.tfrecord'],
                                     height=size, width=size, nclass=10, cache=size <= 64)])

    return d


DATASETS = create_datasets
