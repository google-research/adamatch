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
from typing import Optional, Dict, Iterable, List, Tuple, Callable

import jax
import numpy as np
import objax
from jax import numpy as jn
from objax.jaxboard import SummaryWriter, Summary
from objax.util import EasyDict
from tqdm import tqdm
from tqdm.std import trange

from shared.data import core


class TrainableModule(objax.Module):
    model: objax.Module
    model_ema: objax.Module
    eval_op: Callable
    train_op: Callable

    def __init__(self, nclass: int, params: dict):
        self.nclass = nclass
        self.params = EasyDict(params)

    def __str__(self):
        text = [str(self.model.vars()), 'Parameters'.center(79, '-')]
        for kv in sorted(self.params.items()):
            text.append('%-32s %s' % kv)
        return '\n'.join(text)

    def train_step(self, summary: objax.jaxboard.Summary, data: dict, step: int):
        kv = self.train_op(step, data['image'], data['label'])
        for k, v in kv.items():
            if jn.isnan(v):
                raise ValueError('NaN', k)
            summary.scalar(k, float(v))

    def eval(self, summary: Summary, epoch: int, test: Dict[str, Iterable], valid: Optional[Iterable] = None):
        def get_accuracy(dataset: Iterable):
            accuracy, total, batch = 0, 0, None
            for data in tqdm(dataset, leave=False, desc='Evaluating'):
                x, y = data['image'].numpy(), data['label'].numpy()
                total += x.shape[0]
                batch = batch or x.shape[0]
                if x.shape[0] != batch:
                    # Pad the last batch if it's smaller than expected (must divide properly on GPUs).
                    x = np.concatenate([x] + [x[-1:]] * (batch - x.shape[0]))
                p = self.eval_op(x)[:y.shape[0]]
                accuracy += (np.argmax(p, axis=1) == data['label'].numpy()).sum()
            return accuracy / total if total else 0

        if valid:
            valid_accuracy = get_accuracy(valid)
            summary.scalar('accuracy/valid', 100 * valid_accuracy)
        else:
            valid_accuracy = 0
        test_accuracy = {key: get_accuracy(value) for key, value in test.items()}
        to_print = []
        for key, value in sorted(test_accuracy.items()):
            summary.scalar(f'accuracy/{key}', 100 * value)
            to_print.append('Acccuracy/%s %.2f' % (key, summary[f'accuracy/{key}']()))
        print(f'Epoch {epoch + 1:%-4d}  Loss {summary["losses/xe"]():%.2f}  '
              f'{" ".join(to_print)} (Valid {valid_accuracy * 100:%.2f})')

    def auto_gpu(self):
        if isinstance(self.train_op, objax.Parallel):
            return self.train_op.vars().replicate()
        return objax.util.dummy_context_mgr()

    def train(self, train_kimg: int, report_kimg: int, train: Iterable,
              test: Dict[str, Iterable], logdir: str, keep_ckpts: int, verbose: bool = True):
        if verbose:
            print(self)
            print()
            print('Training config'.center(79, '-'))
            print('%-20s %s' % ('Project:', os.path.basename(core.DATA_DIR)))
            print('%-20s %s' % ('Test sets:', test.keys()))
            print('%-20s %s' % ('Work directory:', logdir))
            print()
        ckpt = objax.io.Checkpoint(logdir=logdir, keep_ckpts=keep_ckpts)
        start_epoch = ckpt.restore(self.vars())[0]

        tensorboard = SummaryWriter(os.path.join(logdir, 'tb'))
        train_iter = iter(train)
        for epoch in range(start_epoch, train_kimg // report_kimg):
            summary = Summary()
            loop = trange(0, report_kimg << 10, self.params.batch,
                          leave=False, unit='img', unit_scale=self.params.batch,
                          desc='Epoch %d/%d' % (1 + epoch, train_kimg // report_kimg))
            with self.auto_gpu():
                for step in loop:
                    self.train_step(summary, next(train_iter), step=np.uint32(step + (epoch * (report_kimg << 10))))

                self.eval(summary, epoch, test)
            tensorboard.write(summary, step=(epoch + 1) * report_kimg * 1024)
            ckpt.save(self.vars(), epoch + 1)


class ScheduleCosPhases:
    def __init__(self, v: float, phases: List[Tuple[float, float]], start_value: float = 1):  # (Phase end, value)
        self.v = v
        self.start_value = float(start_value)
        self.phases = tuple(tuple(map(float, x)) for x in phases)

    def __call__(self, progress: float):
        start, gain, cur_gain = 0., self.start_value, self.start_value
        for stop, next_gain in self.phases:
            assert stop > start
            scale = 0.5 - 0.5 * jn.cos(jn.pi * (progress - start) / (stop - start))
            gain = jax.lax.cond(jn.logical_and(jn.greater_equal(progress, start), jn.less_equal(progress, stop)),
                                lambda gain: cur_gain + (next_gain - cur_gain) * scale,
                                lambda gain: gain,
                                operand=gain)
            cur_gain, start = next_gain, stop
        return self.v * gain


class ScheduleCos:
    def __init__(self, v: float, decay: float):
        self.v = float(v)
        self.slope = jn.arccos(decay)

    def __call__(self, progress: float):
        return self.v * jn.cos(self.slope * progress)
