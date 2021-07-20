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
import os
from typing import Callable, Optional, Tuple, List, Iterable

import numpy as np
import tensorflow as tf
from absl import app
from absl import flags
from absl.flags import FLAGS

# Data directory. Value is initialized in _data_setup
#
# Note that if you need to use DATA_DIR outside of this module then
# you should do following:
#     from shared.data import core as shared_data
#     ...
#     dir = shared_data.DATA_DIR
#
# If you directly import DATA_DIR:
#   from shared.data.core import DATA_DIR
# then None will be imported.
DATA_DIR = None

_DATA_CACHE = None

flags.DEFINE_integer('para_parse', 8, 'Parallel parsing.')
flags.DEFINE_integer('para_augment', 8, 'Parallel augmentation.')
flags.DEFINE_integer('shuffle', 8192, 'Size of dataset shuffling.')
flags.DEFINE_string('data_dir', '',
                    'Data directory. '
                    'If None then environment variable ML_DATA '
                    'will be used as a data directory.')

DATA_DIR = os.path.join(os.environ['ML_DATA'], 'NextMatch') if 'ML_DATA' in os.environ else ''


def _data_setup():
    # set up data directory
    global DATA_DIR
    if FLAGS.data_dir != '':
        DATA_DIR = os.path.join(FLAGS.data_dir, 'NextMatch')
    assert DATA_DIR


app.call_after_init(_data_setup)


def from_uint8(image):
    return (tf.cast(image, tf.float32) - 127.5) / 128


def to_uint8(image):
    return tf.cast(128 * image + 127.5, tf.uint8)


def label_parse(index: int, record: str, *args):
    features = tf.io.parse_single_example(record, features={'label': tf.io.FixedLenFeature([], tf.int64)})
    return dict(index=index, label=features['label'])


def record_parse(index: int, record: str, image_shape: Tuple[int, int, int]):
    features = tf.io.parse_single_example(record, features={'image': tf.io.FixedLenFeature([], tf.string),
                                                            'label': tf.io.FixedLenFeature([], tf.int64)})
    image = tf.image.decode_image(features['image'])
    image.set_shape(image_shape)
    image = from_uint8(image)
    return dict(index=index, image=image, label=features['label'])


def record_parse_mnist(index: int, record: str, image_shape: Tuple[int, int, int]):
    del image_shape
    features = tf.io.parse_single_example(record, features={'image': tf.io.FixedLenFeature([], tf.string),
                                                            'label': tf.io.FixedLenFeature([], tf.int64)})
    image = tf.image.decode_image(features['image'])
    image = tf.pad(image, [(2, 2), (2, 2), (0, 0)])
    image.set_shape((32, 32, 3))
    image = from_uint8(image)
    return dict(index=index, image=image, label=features['label'])


class DataSet:
    """Wrapper for tf.data.Dataset to permit extensions."""

    def __init__(self, data: tf.data.Dataset,
                 image_shape: Tuple[int, int, int],
                 parse_fn: Optional[Callable] = record_parse):
        self.data = data
        self.parse_fn = parse_fn
        self.image_shape = image_shape

    @classmethod
    def from_arrays(cls, images: np.ndarray, labels: np.ndarray):
        return cls(tf.data.Dataset.from_tensor_slices(dict(image=images, label=labels,
                                                           index=np.arange(images.shape[0], dtype=np.int64))),
                   images.shape[1:], parse_fn=None)

    @classmethod
    def from_files(cls, filenames: List[str],
                   image_shape: Tuple[int, int, int],
                   parse_fn: Optional[Callable] = record_parse,
                   cache: bool = False):
        filenames_in = filenames
        filenames = sorted(sum([tf.io.gfile.glob(x) for x in filenames], []))
        if not filenames:
            raise ValueError('Empty dataset, files not found:', filenames_in)
        d = tf.data.TFRecordDataset(filenames, num_parallel_reads=len(filenames))
        d = d.cache() if cache else d
        return cls(d.enumerate(), image_shape, parse_fn=parse_fn)

    @classmethod
    def from_tfds(cls, dataset: tf.data.Dataset, image_shape: Tuple[int, int, int], cache: bool = False):
        d = dataset.cache() if cache else dataset.prefetch(16384)
        d = d.enumerate()
        d = d.map(lambda index, x: dict(image=tf.cast(x['image'], tf.float32) / 127.5 - 1, label=x['label'], index=x))
        return cls(d, image_shape, parse_fn=None)

    def __iter__(self):
        return iter(self.data)

    def __getattr__(self, item):
        if item in self.__dict__:
            return self.__dict__[item]

        def call_and_update(*args, **kwargs):
            v = getattr(self.__dict__['data'], item)(*args, **kwargs)
            if isinstance(v, tf.data.Dataset):
                return self.__class__(v, self.image_shape, parse_fn=self.parse_fn)
            return v

        return call_and_update

    def dmap(self, f: Callable, para: int = 0):
        return self.map(lambda x: f(**x), para or FLAGS.para_parse)

    def nchw(self, key: str = 'image'):
        def apply(d_in):
            return {**d_in, key: tf.transpose(d_in[key], [0, 3, 1, 2])}

        return self.map(apply, FLAGS.para_parse)

    def one_hot(self, nclass):
        return self.dmap(lambda label, **kw: dict(label=tf.one_hot(label, nclass), **kw))

    def parse(self, para: int = 0):
        if not self.parse_fn:
            return self
        para = para or FLAGS.para_parse
        if self.image_shape:
            return self.map(lambda index, record: self.parse_fn(index, record, self.image_shape), para)
        return self.map(lambda index, record: self.parse_fn(index, record), para)

    def __len__(self):
        count = 0
        for _ in self.data:
            count += 1
        return count


class Numpyfier:
    def __init__(self, dataset: Iterable):
        self.dataset = dataset

    def __iter__(self):
        for d in self.dataset:
            yield {k: v.numpy() for k, v in d.items()}
