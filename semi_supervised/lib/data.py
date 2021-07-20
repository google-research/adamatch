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
import collections
import functools
import multiprocessing
from time import sleep
from typing import List, Tuple

import numpy as np
import objax
import tensorflow as tf
from absl.flags import FLAGS

from shared.data.augment import ctaugment
from shared.data.augment.augment import AugmentPoolBase
from shared.data.augment.core import get_tf_augment
from shared.data.augment.ctaugment import CTAugment
from shared.data.core import DataSet
from shared.data.core import Numpyfier
from shared.data.merger import DataMerger
from shared.util import StoppableThread


class MixData(AugmentPoolBase):
    def __init__(self,
                 labeled: DataSet,
                 unlabeled: DataSet,
                 nclass: int, batch: int, uratio: int):
        a = FLAGS.augment.split('(')[-1]
        a = a[:-1].split(',')
        weak, strong = tuple(get_tf_augment(ai, size=labeled.image_shape[0]) for ai in a)

        def bi_augment(d):
            return dict(image=tf.stack([weak(d)['image'], strong(d)['image']]), index=d['index'], label=d['label'])

        x, u = (v.repeat().shuffle(FLAGS.shuffle).parse().map(bi_augment, FLAGS.para_augment).nchw()
                for v in (labeled, unlabeled))
        x = Numpyfier(x.batch(batch).one_hot(nclass).prefetch(16))
        u = Numpyfier(u.batch(batch * uratio).prefetch(16))
        self.train = DataMerger((x, u))

    def __iter__(self) -> dict:
        for x, u in self.train:
            yield dict(x=x, u=u)


class CTAData(AugmentPoolBase):
    def __init__(self,
                 labeled: DataSet,
                 unlabeled: DataSet,
                 nclass: int, batch: int, uratio: int):
        a = FLAGS.augment.split('(')[-1]
        a = a[:-1].split(',')
        a, kwargs = a[:2], {k: v for k, v in (x.split('=') for x in a[2:])}
        h, w, c = labeled.image_shape
        para = FLAGS.para_augment
        probe_shape = (para, batch, c, h, w)
        x_shape = (para, batch, 2, c, h, w)
        u_shape = (para, batch * uratio, 2, c, h, w)
        del h, w, c
        self.shapes = probe_shape, x_shape, u_shape
        self.check_mem_requirements()
        self.probe, self.x, self.u = self.get_np_arrays(self.shapes)
        self.pool = multiprocessing.Pool(para)
        self.probe_period = int(kwargs.get('probe', 1))
        self.cta = CTAugment(int(kwargs.get('depth', 2)),
                             float(kwargs.get('th', 0.8)),
                             float(kwargs.get('decay', 0.99)))
        self.to_schedule = collections.deque()
        self.deque = collections.deque()
        self.free = list(range(para))
        self.thread = StoppableThread(target=self.scheduler)
        self.thread.start()
        weak, strong = tuple(get_tf_augment(ai, size=labeled.image_shape[0]) for ai in a)

        def bi_augment(d):
            return dict(image=tf.stack([weak(d)['image'], strong(d)['image']]), index=d['index'], label=d['label'])

        x, u = (v.repeat().shuffle(FLAGS.shuffle).parse().map(bi_augment, FLAGS.para_augment).nchw()
                for v in (labeled, unlabeled))
        x = Numpyfier(x.batch(batch).one_hot(nclass).prefetch(16))
        u = Numpyfier(u.batch(batch * uratio).prefetch(16))
        self.train = DataMerger((x, u))

    def stop(self):
        self.thread.stop()

    def scheduler(self):
        while not self.thread.stopped():
            sleep(0.0001)
            while self.to_schedule:
                d, idx, do_probe = self.to_schedule.popleft()
                self.x[idx] = d['x']['image']
                self.u[idx] = d['u']['image']
                worker_args = (idx, self.shapes, do_probe, self.cta)
                self.deque.append((d, idx, self.pool.apply_async(self.worker, worker_args)))

    @staticmethod
    def worker(idx: int, shapes: List[Tuple[int, ...]], do_probe: bool, cta: CTAugment):
        def cutout_policy():
            return cta.policy(probe=False) + [ctaugment.OP('cutout', (1,))]

        probe, x_array, u_array = (arr[idx] for arr in CTAData.get_np_arrays(shapes))
        x = objax.util.image.nhwc(x_array)
        u = objax.util.image.nhwc(u_array)
        nchw = objax.util.image.nchw
        sx_strong = np.stack([ctaugment.apply(x[i, 1], cutout_policy()) for i in range(x.shape[0])])
        tu_strong = np.stack([ctaugment.apply(u[i, 1], cutout_policy()) for i in range(u.shape[0])])
        x_array[:] = nchw(np.stack([x[:, 0], sx_strong], axis=1))
        u_array[:] = nchw(np.stack([u[:, 0], tu_strong], axis=1))
        if not do_probe:
            return

        policy_list = [cta.policy(probe=True) for _ in range(x.shape[0])]
        probe[:] = nchw(np.stack([ctaugment.apply(x[i, 0], policy)
                                  for i, policy in enumerate(policy_list)]))
        return policy_list

    def update_rates(self, policy, label, y_probe):
        w1 = 1 - 0.5 * np.abs(y_probe - label).sum(1)
        for p in range(w1.shape[0]):
            self.cta.update_rates(policy[p], w1[p])

    def __iter__(self):
        for i, (x, u) in enumerate(self.train):
            if not self.free:
                while not self.deque:
                    sleep(0.0001)
                d, idx, pd = self.deque.popleft()
                self.free.append(idx)
                policy = pd.get()
                if policy:
                    d['probe'] = np.copy(self.probe[idx])
                    d['policy'] = policy
                    d['probe_callback'] = functools.partial(self.update_rates, d['policy'], d['x']['label'])

                d['x']['image'][:] = self.x[idx]
                d['u']['image'][:] = self.u[idx]
                yield d

            self.to_schedule.append((dict(x=x, u=u), self.free.pop(), (i % self.probe_period) == 0))
