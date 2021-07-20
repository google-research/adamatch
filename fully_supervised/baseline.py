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
"""Fully supervised on a pair of domains.
"""
import os
import sys
from typing import Callable

import jax
import jax.numpy as jn
import objax
from absl import app
from absl import flags
from absl.flags import FLAGS
from objax.typing import JaxArray

from fully_supervised.lib.data import MixData, CTAData
from fully_supervised.lib.train import TrainableFSLModule
from shared.data.fsl import DATASETS as FSL_DATASETS
from shared.train import ScheduleCos
from shared.util import setup_tf, MyParallel
from shared.zoo.models import network, ARCHS


class Baseline(TrainableFSLModule):
    def __init__(self, nclass: int, model: Callable, **kwargs):
        super().__init__(nclass, kwargs)
        self.model: objax.Module = model(colors=3, nclass=nclass, **kwargs)
        self.model_ema = objax.optimizer.ExponentialMovingAverageModule(self.model, momentum=0.999)
        train_vars = self.model.vars()
        self.opt = objax.optimizer.Momentum(train_vars)
        self.lr = ScheduleCos(self.params.lr, self.params.lr_decay)

        @objax.Function.with_vars(self.model_ema.vars())
        def eval_op(x: JaxArray) -> JaxArray:
            return objax.functional.softmax(self.model_ema(x, training=False))

        def loss_function(sx, sy, tx, ty):
            c, h, w = sx.shape[-3:]
            xu = jn.concatenate((sx, tx)).reshape((-1, c, h, w))
            logit = self.model(xu, training=True)
            logit_sx, logit_tx = jn.split(logit, (2 * sx.shape[0],))
            logit_sx_weak, logit_sx_strong = logit_sx[::2], logit_sx[1::2]
            logit_tx_weak, logit_tx_strong = logit_tx[::2], logit_tx[1::2]

            xe = 0.25 * (objax.functional.loss.cross_entropy_logits(logit_sx_weak, sy).mean() +
                         objax.functional.loss.cross_entropy_logits(logit_sx_strong, sy).mean() +
                         objax.functional.loss.cross_entropy_logits(logit_tx_weak, ty).mean() +
                         objax.functional.loss.cross_entropy_logits(logit_tx_strong, ty).mean())
            wd = 0.5 * sum((v.value ** 2).sum() for k, v in train_vars.items() if k.endswith('.w'))

            loss = xe + self.params.wd * wd
            return loss, {'losses/xe': xe, 'losses/wd': wd}

        gv = objax.GradValues(loss_function, train_vars)

        @objax.Function.with_vars(self.vars())
        def train_op(step, sx, sy, tx, ty, probe=None):
            y_probe = eval_op(probe) if probe is not None else None
            p = step / (FLAGS.train_mimg << 20)
            lr = self.lr(p)
            g, v = gv(sx, sy, tx, ty)
            self.opt(lr, objax.functional.parallel.pmean(g))
            self.model_ema.update_ema()
            return objax.functional.parallel.pmean({'monitors/lr': lr, **v[1]}), y_probe

        self.train_op = MyParallel(train_op, reduce=lambda x: x)
        self.eval_op = MyParallel(eval_op)


def main(argv):
    del argv
    print('JAX host: %d / %d' % (jax.host_id(), jax.host_count()))
    print('JAX devices:\n%s' % '\n'.join(str(d) for d in jax.devices()), flush=True)
    setup_tf()
    # In fully supervised, source and target are both labeled, we just kept the same names to reuse the code.
    if FLAGS.target == '':
        FLAGS.target = FLAGS.source
    source = FSL_DATASETS()[f'{FLAGS.dataset}_{FLAGS.source}-0']()
    target = FSL_DATASETS()[f'{FLAGS.dataset}_{FLAGS.target}-0']()
    module = Baseline(source.nclass, network(FLAGS.arch),
                      lr=FLAGS.lr,
                      lr_decay=FLAGS.lr_decay,
                      wd=FLAGS.wd,
                      arch=FLAGS.arch,
                      batch=FLAGS.batch)
    logdir = f'FSL/{FLAGS.dataset}/{FLAGS.source}/{FLAGS.target}/{FLAGS.augment}/{module.__class__.__name__}/%s' % (
        '_'.join(sorted('%s%s' % k for k in module.params.items())))
    logdir = os.path.join(FLAGS.logdir, logdir)
    test = {
        f'{FLAGS.source}_to_{FLAGS.target}': (list(target.test.values())[0].parse().batch(FLAGS.batch).nchw()
                                              .map(lambda d: {**d, 'domain': 0}).prefetch(16)),
        f'{FLAGS.target}_to_{FLAGS.source}': (list(source.test.values())[0].parse().batch(FLAGS.batch).nchw()
                                              .map(lambda d: {**d, 'domain': 1}).prefetch(16)),
    }
    if FLAGS.test_extra:
        for ds_name in FLAGS.test_extra.split(','):
            ds = FSL_DATASETS()[f'{FLAGS.dataset}_{ds_name}-0']()
            test[f'{FLAGS.target}_to_{ds_name}'] = (list(ds.test.values())[0].parse().batch(FLAGS.batch).nchw()
                                                    .map(lambda d: {**d, 'domain': 1}).prefetch(16))
    if FLAGS.augment.startswith('('):
        train = MixData(source.train, target.train, source.nclass, FLAGS.batch)
    elif FLAGS.augment.startswith('CTA('):
        train = CTAData(source.train, target.train, source.nclass, FLAGS.batch)
    else:
        raise ValueError(f'Augment flag value {FLAGS.augment} not supported.')
    module.train(FLAGS.train_mimg << 10, FLAGS.report_kimg, train, test, logdir, FLAGS.keep_ckpts)
    train.stop()


if __name__ == '__main__':
    flags.DEFINE_enum('arch', 'wrn28-2', ARCHS, 'Model architecture.')
    flags.DEFINE_float('lr', 0.03, 'Learning rate.')
    flags.DEFINE_float('lr_decay', 0.25, 'Learning rate decay.')
    flags.DEFINE_float('wd', 0.001, 'Weight decay.')
    flags.DEFINE_integer('batch', 64, 'Batch size')
    flags.DEFINE_integer('report_kimg', 64, 'Reporting period in kibi-images.')
    flags.DEFINE_integer('train_mimg', 8, 'Training duration in mega-images.')
    flags.DEFINE_integer('keep_ckpts', 5, 'Number of checkpoints to keep (0 for all).')
    flags.DEFINE_string('logdir', 'experiments', 'Directory where to save checkpoints and tensorboard data.')
    flags.DEFINE_string('dataset', 'domainnet32', 'Source data to train on.')
    flags.DEFINE_string('source', 'clipart', 'Source data to train on.')
    flags.DEFINE_string('target', '', 'Target data to train on.')
    flags.DEFINE_string('test_extra', 'clipart,infograph,quickdraw,real,sketch,painting',
                        'Comma-separated list of datasets on which to report test accuracy.')
    FLAGS.set_default('augment', 'CTA(sm,sm,probe=1)')
    FLAGS.set_default('para_augment', 8)
    app.run(main)
