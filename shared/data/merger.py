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
from typing import Iterable

import numpy as np


class DataMerger:
    def __init__(self, datasets: Iterable[Iterable]):
        self.datasets = list(datasets)

    def __iter__(self):
        iterators = [iter(x) for x in self.datasets]
        while 1:
            yield [next(x) for x in iterators]


class MergeBalanced(DataMerger):
    def __init__(self, datasets: Iterable[Iterable], sizes: Iterable[int]):
        del sizes
        super().__init__(datasets)

    def __iter__(self):
        count = len(self.datasets)
        iterators = [iter(x) for x in self.datasets]
        while 1:
            data = [next(x) for x in iterators]
            source = np.zeros([count, data[0]['index'].shape[0]], np.uint32)
            source += np.arange(count, dtype=np.uint32)[:, None]
            mixed = {k: np.concatenate([v[k] for v in data]) for k in ('index', 'label', 'image')}
            mixed['source'] = source.ravel()
            for x in range(len(data)):
                yield {k: v[x::len(data)] for k, v in mixed.items()}


class MergeProportional(DataMerger):
    def __init__(self, datasets: Iterable[Iterable], sizes: Iterable[int]):
        super().__init__(datasets)
        self.ratios = np.array(list(sizes), 'f')
        self.ratios /= self.ratios.sum()
        assert len(self.datasets) == len(self.ratios)

    @staticmethod
    def collect(buffer, iterator, count):
        available = buffer['index'].shape[0]
        if available < count:
            entry = next(iterator)
            for k in buffer:
                buffer[k] = np.concatenate([buffer[k], entry[k]])
        vout = {}
        for k, v in buffer.items():
            vout[k] = v[:count]
            buffer[k] = v[count:]
        return vout

    def __iter__(self):
        count = len(self.datasets)
        p = np.zeros(count, np.uint32)
        iterators = [iter(x) for x in self.datasets]
        buffers = [next(x) for x in iterators]
        batch = buffers[0]['index'].shape[0]
        order = np.arange(batch, dtype=np.uint32)
        while True:
            fsizes = self.ratios * (p.sum() + batch) - p
            sizes = fsizes.astype(np.uint32)
            delta = batch - int(sizes.sum())
            if delta:
                high_p = np.argsort(fsizes - sizes)[-delta:]
                sizes[high_p] += 1
            data = [self.collect(buffers[i], iterators[i], n) for i, n in enumerate(sizes)]
            data = {k: np.concatenate([d[k] for d in data]) for k in data[0]}
            data['source'] = np.concatenate([np.zeros(s, dtype=np.uint32) + i for i, s in enumerate(sizes)])
            np.random.shuffle(order)
            yield {k: v[order] for k, v in data.items()}
