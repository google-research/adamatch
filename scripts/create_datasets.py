#!/usr/bin/env python

# Copyright 2018 Google LLC
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

"""Script to download all datasets and create .tfrecord files.
"""

import collections
import gzip
import itertools
import os
import tarfile
import tempfile
import zipfile
from functools import partial, reduce
from urllib import request
import wget
import h5py

import numpy as np
import scipy.io
import tensorflow as tf
from PIL import Image
from absl import app
from google_drive_downloader import GoogleDriveDownloader as gdd
from objax.util.image import to_png
from tqdm import trange, tqdm

from shared.data import core as libml_data

if 'NEXTMATCH_DOWNLOAD_PATH' in os.environ:
    DOWNLOAD_DIR = os.environ['NEXTMATCH_DOWNLOAD_PATH']
else:
    DOWNLOAD_DIR = os.path.join(libml_data.DATA_DIR, 'Downloads')

URLS = {
    'cifar10': 'https://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz',
    'cifar100': 'https://www.cs.toronto.edu/~kriz/cifar-100-matlab.tar.gz',
    'domainnet': {
        'clipart': 'http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/%sclipart%s',
        'infograph': 'http://csr.bu.edu/ftp/visda/2019/multi-source/%sinfograph%s',
        'painting': 'http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/%spainting%s',
        'quickdraw': 'http://csr.bu.edu/ftp/visda/2019/multi-source/%squickdraw%s',
        'real': 'http://csr.bu.edu/ftp/visda/2019/multi-source/%sreal%s',
        'sketch': 'http://csr.bu.edu/ftp/visda/2019/multi-source/%ssketch%s'
    },
    'mnist': 'http://yann.lecun.com/exdb/mnist/{}',
    'office31': dict(images='0B4IapRTv9pJ1WGZVd1VDMmhwdlE'),
    'svhn': 'http://ufldl.stanford.edu/housenumbers/{}_32x32.mat',
    'mnistm': 'https://www.dropbox.com/s/rb7pr65fo26h9lh/mnist_m.tar.gz?dl=1',
    'syndigit': 'https://storage.googleapis.com/kihyuks-0001/SynDigits/synth_{}_32x32.mat',
    'usps': 'https://storage.googleapis.com/kihyuks-0001/usps.h5',
}


def _encode_png(images):
    return [to_png(images[x]) for x in trange(images.shape[0], desc='PNG Encoding', leave=False)]


def _image_resize(x, size: int):
    """Resizing that tries to minimize artifacts."""
    original = max(x.size)
    if original < size:
        return x.resize((size, size), Image.BICUBIC)
    nearest = original - (original % size)
    if nearest != original:
        x = x.resize((nearest, nearest), Image.BILINEAR)
    if nearest != size:
        x = x.resize((size, size), Image.BOX)
    if x.size[0] != x.size[1]:
        x = x.resize((size, size), Image.BICUBIC)
    return x


def _load_cifar10():
    def unflatten(images):
        return np.transpose(images.reshape((images.shape[0], 3, 32, 32)), [0, 2, 3, 1])

    with tempfile.NamedTemporaryFile() as f:
        request.urlretrieve(URLS['cifar10'], f.name)
        tar = tarfile.open(fileobj=f)
        train_data_batches, train_data_labels = [], []
        for batch in range(1, 6):
            data_dict = scipy.io.loadmat(tar.extractfile('cifar-10-batches-mat/data_batch_{}.mat'.format(batch)))
            train_data_batches.append(data_dict['data'])
            train_data_labels.append(data_dict['labels'].flatten())

        train_set = {'images': np.concatenate(train_data_batches, axis=0),
                     'labels': np.concatenate(train_data_labels, axis=0)}
        data_dict = scipy.io.loadmat(tar.extractfile('cifar-10-batches-mat/test_batch.mat'))
        test_set = {'images': data_dict['data'],
                    'labels': data_dict['labels'].flatten()}
    train_set['images'] = _encode_png(unflatten(train_set['images']))
    test_set['images'] = _encode_png(unflatten(test_set['images']))
    return dict(train=train_set, test=test_set)


def _load_cifar100():
    def unflatten(images):
        return np.transpose(images.reshape((images.shape[0], 3, 32, 32)), [0, 2, 3, 1])

    with tempfile.NamedTemporaryFile() as f:
        request.urlretrieve(URLS['cifar100'], f.name)
        tar = tarfile.open(fileobj=f)
        data_dict = scipy.io.loadmat(tar.extractfile('cifar-100-matlab/train.mat'))
        train_set = {'images': data_dict['data'],
                     'labels': data_dict['fine_labels'].flatten()}
        data_dict = scipy.io.loadmat(tar.extractfile('cifar-100-matlab/test.mat'))
        test_set = {'images': data_dict['data'],
                    'labels': data_dict['fine_labels'].flatten()}
    train_set['images'] = _encode_png(unflatten(train_set['images']))
    test_set['images'] = _encode_png(unflatten(test_set['images']))
    return dict(train=train_set, test=test_set)


def _load_domainnet(domain: str, size: int) -> dict:
    assert domain in ('clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch')
    path = os.path.join(DOWNLOAD_DIR, 'DomainNet')
    os.makedirs(path, exist_ok=True)
    prefixes = '', 'txt/', 'txt/'
    suffixes = '.zip', '_train.txt', '_test.txt'
    files = [os.path.join(path, f'{domain}{suffix}') for suffix in suffixes]
    for f, prefix, suffix in zip(files, prefixes, suffixes):
        if not os.path.exists(f):
            print(f'Downloading {URLS["domainnet"][domain] % (prefix, suffix)}')
            request.urlretrieve(URLS['domainnet'][domain] % (prefix, suffix), f)

    train = [(k, int(v)) for k, v in [x.split() for x in open(files[1], 'r').readlines()]]
    test = [(k, int(v)) for k, v in [x.split() for x in open(files[2], 'r').readlines()]]
    zipped = zipfile.ZipFile(files[0])
    image = {}
    for info in tqdm(zipped.infolist(), 'Resizing images', leave=False):
        if info.is_dir():
            continue
        with zipped.open(info) as f:
            x = np.array(_image_resize(Image.open(f), size))
            image[info.filename] = to_png(x)

    np.random.seed(0)
    np.random.shuffle(train)
    return dict(all=dict(images=[image[k] for k, _ in train + test], labels=np.array([v for _, v in train + test])),
                test=dict(images=[image[k] for k, _ in test], labels=np.array([v for _, v in test])),
                train=dict(images=[image[k] for k, _ in train], labels=np.array([v for _, v in train])))


def _load_mnist():
    image_filename = '{}-images-idx3-ubyte.gz'
    label_filename = '{}-labels-idx1-ubyte.gz'
    split_files = [('train', 'train'), ('test', 't10k')]
    splits = {}
    for split, split_file in split_files:
        with tempfile.NamedTemporaryFile() as f:
            url = URLS['mnist'].format(image_filename.format(split_file))
            print(url)
            request.urlretrieve(url, f.name)
            with gzip.GzipFile(fileobj=f, mode='r') as data:
                assert _read32(data) == 2051
                n_images = _read32(data)
                row = _read32(data)
                col = _read32(data)
                images = np.frombuffer(data.read(n_images * row * col), dtype=np.uint8)
                images = images.reshape((n_images, row, col, 1))
        with tempfile.NamedTemporaryFile() as f:
            request.urlretrieve(URLS['mnist'].format(label_filename.format(split_file)), f.name)
            with gzip.GzipFile(fileobj=f, mode='r') as data:
                assert _read32(data) == 2049
                n_labels = _read32(data)
                labels = np.frombuffer(data.read(n_labels), dtype=np.uint8)
        splits[split] = {'images': _encode_png(images), 'labels': labels}
    return splits


def _load_mnist32():
    image_filename = '{}-images-idx3-ubyte.gz'
    label_filename = '{}-labels-idx1-ubyte.gz'
    split_files = [('train', 'train'), ('test', 't10k')]
    splits = {}
    for split, split_file in split_files:
        with tempfile.NamedTemporaryFile() as f:
            url = URLS['mnist'].format(image_filename.format(split_file))
            print(url)
            request.urlretrieve(url, f.name)
            with gzip.GzipFile(fileobj=f, mode='r') as data:
                assert _read32(data) == 2051
                n_images = _read32(data)
                row = _read32(data)
                col = _read32(data)
                images = np.frombuffer(data.read(n_images * row * col), dtype=np.uint8)
                images = images.reshape((n_images, row, col, 1))
                # Pad 2x2 so that it becomes 32x32
                images_pad = np.zeros((images.shape[0],
                                       images.shape[1] + 4,
                                       images.shape[2] + 4,
                                       images.shape[3])).astype(np.uint8)
                images_pad[:, 2:-2, 2:-2, :] = images
        with tempfile.NamedTemporaryFile() as f:
            request.urlretrieve(URLS['mnist'].format(label_filename.format(split_file)), f.name)
            with gzip.GzipFile(fileobj=f, mode='r') as data:
                assert _read32(data) == 2049
                n_labels = _read32(data)
                labels = np.frombuffer(data.read(n_labels), dtype=np.uint8)
        splits[split] = {'images': _encode_png(images_pad), 'labels': labels}
    return splits


def _load_mnistm():
    with tempfile.NamedTemporaryFile() as f:
        request.urlretrieve(URLS['mnistm'], f.name)
        tar = tarfile.open(fileobj=f)
        splits = {}
        for split in ['train', 'test']:
            prefix = f'mnist_m/mnist_m_{split}'
            img_list = tar.extractfile(f'{prefix}_labels.txt').readlines()
            images = []
            labels = []
            for img_path in tqdm(img_list, f'Loading mnistm {split} images and labels', leave=False):
                images.append(np.array(Image.open(tar.extractfile(os.path.join(
                              prefix, img_path.split()[0].decode('utf-8'))))))
                labels.append(int(img_path.split()[1].decode('utf-8')))
            images = np.stack(images, axis=0)
            splits[split] = {'images': _encode_png(images), 'labels': labels}
    return splits


def _load_syndigit():
    splits = {}
    for split in ['train', 'test']:
        filename = 'synth_{}_32x32.mat'.format(split)
        if not os.path.exists(filename):
            wget.download(URLS['syndigit'].format(split), out=filename)
        data_dict = scipy.io.loadmat(filename)
        images = np.transpose(data_dict['X'], (3, 0, 1, 2))
        labels = data_dict['y'].flatten()
        splits[split] = {'images': _encode_png(images), 'labels': labels}
    return splits

def _load_usps():
    def _hdf5(path, data_key = "data", target_key = "target", flatten = True):
        """
            loads data from hdf5:
            - hdf5 should have 'train' and 'test' groups
            - each group should have 'data' and 'target' dataset or spcify the key
            - flatten means to flatten images N * (C * H * W) as N * D array
            code from: https://www.kaggle.com/bistaumanga/usps-getting-started?scriptVersionId=3215146&cellId=3
        """
        with h5py.File(path, 'r') as hf:
            train = hf.get('train')
            X_tr = train.get(data_key)[:]
            y_tr = train.get(target_key)[:]
            test = hf.get('test')
            X_te = test.get(data_key)[:]
            y_te = test.get(target_key)[:]
            if flatten:
                X_tr = X_tr.reshape(X_tr.shape[0], reduce(lambda a, b: a * b, X_tr.shape[1:]))
                X_te = X_te.reshape(X_te.shape[0], reduce(lambda a, b: a * b, X_te.shape[1:]))
        return X_tr, y_tr, X_te, y_te

    filename = 'usps.h5'
    if not os.path.exists(filename):
        wget.download(URLS['usps'], out=filename)
    X_tr, y_tr, X_te, y_te = _hdf5(filename)
    X_tr = np.concatenate([(255.0 * X_tr).astype(np.uint8).reshape(-1, 16, 16, 1)] * 3, axis=-1)
    X_tr = np.stack([np.array(_image_resize(Image.fromarray(x), 32)) for x in X_tr], axis=0)
    X_te = np.concatenate([(255.0 * X_te).astype(np.uint8).reshape(-1, 16, 16, 1)] * 3, axis=-1)
    X_te = np.stack([np.array(_image_resize(Image.fromarray(x), 32)) for x in X_te], axis=0)
    splits = {'train': {'images': _encode_png(X_tr), 'labels': y_tr},
              'test': {'images': _encode_png(X_te), 'labels': y_te}}
    return splits


def _load_digitfive(domain: str, size: int) -> dict:
    assert size == 32
    assert domain in 'mnist svhn usps mnistm syndigit'.split()
    if domain == 'mnist':
        return _load_mnist32()
    elif domain == 'svhn':
        return _load_svhn()
    elif domain == 'usps':
        return _load_usps()
    elif domain == 'mnistm':
        return _load_mnistm()
    elif domain == 'syndigit':
        return _load_syndigit()


def _load_office31(domain: str, size: int) -> dict:
    assert domain in 'amazon dslr webcam'.split()
    path = os.path.join(DOWNLOAD_DIR, 'office31_images.tgz')
    if not os.path.exists(path):
        gdd.download_file_from_google_drive(file_id=URLS['office31']['images'], dest_path=path, overwrite=True)
        if b'Quota exceeded' in open(path, 'rb').read(1024):
            os.remove(path)
            raise FileNotFoundError('Quota exceeded: File office31_images.tgz for Office31 could not be downloaded from'
                                    ' Google drive. Try again later.')
    data = collections.defaultdict(list)
    with tarfile.open(name=path, mode='r:gz') as tar:
        for entry in tar.getmembers():
            domain_, _, class_, name = entry.name.split('/')
            if domain == domain_:
                data[class_].append((class_, name, entry))

        np.random.seed(0)
        train, test = [], []
        for class_ in data.keys():
            np.random.shuffle(data[class_])
            total_num_frames = len(data[class_])
            num_train_frames = int(0.8*total_num_frames)
            train_frames = data[class_][:num_train_frames]
            test_frames = data[class_][num_train_frames:]
            assert len(train_frames) + len(test_frames) == total_num_frames
            train += train_frames
            test += test_frames

        train_images, train_labels, train_label_set = [], [], set()
        for class_, name, entry in tqdm(train, leave=False, desc='Resizing train images'):
            train_images.append(np.array(_image_resize(Image.open(tar.extractfile(entry)), size)))
            assert train_images[-1].shape == (size, size, 3)
            train_labels.append(class_)
            train_label_set.add(class_)
        train_label_id = {x: p for p, x in enumerate(sorted(train_label_set))}

        test_images, test_labels, test_label_set  =  [], [], set()
        for class_, name, entry in tqdm(test, leave=False, desc='Resizing train images'):
            test_images.append(np.array(_image_resize(Image.open(tar.extractfile(entry)), size)))
            assert test_images[-1].shape == (size, size, 3)
            test_labels.append(class_)
            test_label_set.add(class_)
        test_label_id = {x: p for p, x in enumerate(sorted(test_label_set))}

    return dict(train=dict(images=_encode_png(np.stack(train_images)),
                           labels=np.array([train_label_id[x] for x in train_labels], 'int32')),
                test=dict(images=_encode_png(np.stack(test_images)),
                           labels=np.array([test_label_id[x] for x in test_labels], 'int32')))


def _load_svhn():
    splits = collections.OrderedDict()
    for split in ['train', 'test', 'extra']:
        with tempfile.NamedTemporaryFile() as f:
            request.urlretrieve(URLS['svhn'].format(split), f.name)
            data_dict = scipy.io.loadmat(f.name)
        dataset = {}
        dataset['images'] = np.transpose(data_dict['X'], [3, 0, 1, 2])
        dataset['images'] = _encode_png(dataset['images'])
        dataset['labels'] = data_dict['y'].reshape((-1))
        # SVHN raw data uses labels from 1 to 10; use 0 to 9 instead.
        dataset['labels'] %= 10  # Label number 10 is for 0.
        splits[split] = dataset
    return splits


def _read32(data):
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(data.read(4), dtype=dt)[0]


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _save_as_tfrecord(data, filename):
    assert len(data['images']) == len(data['labels'])
    filename = os.path.join(libml_data.DATA_DIR, filename + '.tfrecord')
    print('Saving dataset:', filename)
    with tf.io.TFRecordWriter(filename) as writer:
        for x in trange(len(data['images']), desc='Building records'):
            feat = dict(image=_bytes_feature(data['images'][x]),
                        label=_int64_feature(data['labels'][x]))
            record = tf.train.Example(features=tf.train.Features(feature=feat))
            writer.write(record.SerializeToString())
    print('Saved:', filename)


def _is_installed(name, checksums):
    for subset, checksum in checksums.items():
        filename = os.path.join(libml_data.DATA_DIR, '%s-%s.tfrecord' % (name, subset))
        if not tf.io.gfile.exists(filename):
            return False
    return True


def _save_files(files, *args, **kwargs):
    del args, kwargs
    for folder in frozenset(os.path.dirname(x) for x in files):
        tf.io.gfile.makedirs(os.path.join(libml_data.DATA_DIR, folder))
    for filename, contents in files.items():
        with tf.io.gfile.GFile(os.path.join(libml_data.DATA_DIR, filename), 'w') as f:
            f.write(contents)


def _is_installed_folder(name, folder):
    return tf.io.gfile.exists(os.path.join(libml_data.DATA_DIR, name, folder))


CONFIGS = {
    'cifar10': dict(loader=_load_cifar10, checksums=dict(train=None, test=None)),
    'cifar100': dict(loader=_load_cifar100, checksums=dict(train=None, test=None)),
    'mnist': dict(loader=_load_mnist, checksums=dict(train=None, test=None)),
    'svhn': dict(loader=_load_svhn, checksums=dict(train=None, test=None, extra=None)),
}
CONFIGS.update({
    f'domainnet{size}_{domain}': dict(loader=partial(_load_domainnet, domain=domain, size=size),
                                      checksums=dict(train=None, test=None, all=None))
    for size, domain in
    itertools.product((32, 64, 128, 224), 'clipart infograph painting quickdraw real sketch'.split())
})
CONFIGS.update({
    f'office31{size}_{domain}': dict(loader=partial(_load_office31, domain=domain, size=size),
                                     checksums=dict(train=None))
    for size, domain in itertools.product((32, 64, 128, 224), 'amazon dslr webcam'.split())
})
CONFIGS.update({
    f'digitfive{size}_{domain}': dict(loader=partial(_load_digitfive, domain=domain, size=size),
                                     checksums=dict(train=None))
    for size, domain in itertools.product((32,), 'mnist svhn usps mnistm syndigit'.split())
})

def main(argv):
    if len(argv[1:]):
        subset = set(argv[1:])
    else:
        subset = set(CONFIGS.keys())
    tf.io.gfile.makedirs(libml_data.DATA_DIR)
    for name in subset:
        assert name in CONFIGS, f'Dataset not recognized {name}'
    for name, config in CONFIGS.items():
        if name not in subset:
            continue
        if 'is_installed' in config:
            if config['is_installed']():
                print('Skipping already installed:', name)
                continue
        elif _is_installed(name, config['checksums']):
            print('Skipping already installed:', name)
            continue
        print('Preparing', name)
        datas = config['loader']()
        saver = config.get('saver', _save_as_tfrecord)
        for sub_name, data in datas.items():
            if sub_name == 'readme':
                filename = os.path.join(libml_data.DATA_DIR, '%s-%s.txt' % (name, sub_name))
                with tf.io.gfile.GFile(filename, 'w') as f:
                    f.write(data)
            elif sub_name == 'files':
                for file_and_data in data:
                    path = os.path.join(libml_data.DATA_DIR, file_and_data.filename)
                    with tf.io.gfile.GFile(path, "wb") as f:
                        f.write(file_and_data.data)
            else:
                saver(data, '%s-%s' % (name, sub_name))


if __name__ == '__main__':
    app.run(main)
