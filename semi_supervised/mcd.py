# Copyright 2019 Google LLC
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
"""FixMatch with Distribution Alignment and Adaptative Confidence Ratio.
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
from objax.functional import softmax
from objax.typing import JaxArray

from semi_supervised.lib.data import MixData, CTAData
from semi_supervised.lib.train import TrainableSSLModule
from shared.data.fsl import DATASETS as FSL_DATASETS
from shared.data.ssl import DATASETS as SSL_DATASETS, DataSetSSL
from shared.train import ScheduleCos
from shared.util import setup_tf, MyParallel
from shared.zoo.models import network, ARCHS


class MCD(TrainableSSLModule):
    def __init__(self, nclass: int, model: Callable, **kwargs):
        super().__init__(nclass, kwargs)
        self.model: objax.Module = model(colors=3, nclass=nclass, **kwargs)
        self.model_ema = objax.optimizer.ExponentialMovingAverageModule(self.model, momentum=0.999)
        if FLAGS.arch.endswith('pretrain'):
            # Initialize weights of EMA with pretrained model's weights.
            self.model_ema.ema.momentum = 0
            self.model_ema.update_ema()
            self.model_ema.ema.momentum = 0.999
        self.c1: objax.Module = self.model[-1]
        self.c2: objax.Module = objax.nn.Linear(self.c1.w.value.shape[0], nclass)
        self.gen: objax.Module = self.model[:-1]
        self.opt1 = objax.optimizer.Momentum(self.gen.vars() + self.c1.vars('c1') + self.c2.vars('c2'))
        self.opt2 = objax.optimizer.Momentum(self.c1.vars('c1') + self.c2.vars('c2'))
        self.opt3 = objax.optimizer.Momentum(self.gen.vars())
        self.lr = ScheduleCos(self.params.lr, self.params.lr_decay)

        @objax.Function.with_vars(self.model_ema.vars())
        def eval_op(x: JaxArray) -> JaxArray:
            return objax.functional.softmax(self.model_ema(x, training=False))

        def get_two_outputs(v):
            feat = self.gen(v, training=True)
            return self.c1(feat), self.c2(feat)

        def loss_function_phase1(x, y):
            x1, x2 = get_two_outputs(x[:, 0])
            xes = (objax.functional.loss.cross_entropy_logits(x1, y).mean() +
                   objax.functional.loss.cross_entropy_logits(x2, y).mean())
            return xes, {'losses/xe': xes}

        def loss_function_phase2(x, u, y):
            saved = self.gen.vars().tensors()
            x1, x2 = get_two_outputs(x[:, 0])
            u1, u2 = get_two_outputs(u[:, 0])
            self.gen.vars().assign(saved)

            xes = (objax.functional.loss.cross_entropy_logits(x1, y).mean() +
                   objax.functional.loss.cross_entropy_logits(x2, y).mean())
            dis = jn.abs(softmax(u1) - softmax(u2)).mean()
            return xes - dis, {'losses/xe2': xes, 'losses/dis2': dis}

        def loss_function_phase3(u):
            u1, u2 = get_two_outputs(u[:, 0])
            dis = jn.abs(softmax(u1) - softmax(u2)).mean()
            return dis, {'losses/dis3': dis}

        gv1 = objax.GradValues(loss_function_phase1, self.gen.vars() + self.c1.vars('c1') + self.c2.vars('c2'))
        gv2 = objax.GradValues(loss_function_phase2, self.c1.vars('c1') + self.c2.vars('c2'))
        gv3 = objax.GradValues(loss_function_phase3, self.gen.vars())

        @objax.Function.with_vars(self.vars())
        def train_op(step, x, y, u, probe=None):
            y_probe = eval_op(probe) if probe is not None else None
            p = step / (FLAGS.train_mimg << 20)
            lr = self.lr(p)
            g, v1 = gv1(x, y)
            self.opt1(lr, objax.functional.parallel.pmean(g))
            g, v2 = gv2(x, u, y)
            self.opt2(lr, objax.functional.parallel.pmean(g))
            v3 = {}
            for _ in range(self.params.wu):
                g, v = gv3(u)
                for k, val in v[1].items():
                    v3[k] = v3.get(k, 0) + val / self.params.wu
                self.opt3(lr, objax.functional.parallel.pmean(g))
            self.model_ema.update_ema()
            return objax.functional.parallel.pmean({'monitors/lr': lr, **v1[1], **v2[1], **v[1]}), y_probe

        self.train_op = MyParallel(train_op, reduce=lambda x: x)
        self.eval_op = MyParallel(eval_op)


def main(argv):
    del argv
    print('JAX host: %d / %d' % (jax.host_id(), jax.host_count()))
    print('JAX devices:\n%s' % '\n'.join(str(d) for d in jax.devices()), flush=True)
    setup_tf()
    dataset_name, samples_per_class, dataset_seed = DataSetSSL.parse_name(f'{FLAGS.dataset}')
    labeled = SSL_DATASETS()[dataset_name](samples_per_class, dataset_seed)
    unlabeled = FSL_DATASETS()[f'{dataset_name}-0']()
    testsets = [unlabeled.test]
    module = MCD(labeled.nclass, network(FLAGS.arch),
                 lr=FLAGS.lr,
                 lr_decay=FLAGS.lr_decay,
                 wd=FLAGS.wd,
                 arch=FLAGS.arch,
                 batch=FLAGS.batch,
                 wu=FLAGS.wu,
                 uratio=FLAGS.uratio)
    logdir = f'SSL/{FLAGS.dataset}/{FLAGS.augment}/{module.__class__.__name__}/%s' % (
        '_'.join(sorted('%s%s' % k for k in module.params.items())))
    logdir = os.path.join(FLAGS.logdir, logdir)
    test = {}
    for domain, testset in enumerate(testsets):
            test.update((k, v.parse().batch(FLAGS.batch).nchw().map(lambda d: {**d, 'domain': domain}).prefetch(16))
                         for k, v in testset.items())
    if FLAGS.augment.startswith('('):
        train = MixData(labeled.train, unlabeled.train, labeled.nclass, FLAGS.batch, FLAGS.uratio)
    elif FLAGS.augment.startswith('CTA('):
        train = CTAData(labeled.train, unlabeled.train, labeled.nclass, FLAGS.batch, FLAGS.uratio)
    else:
        raise ValueError(f'Augment flag value {FLAGS.augment} not supported.')
    module.train(FLAGS.train_mimg << 10, FLAGS.report_kimg, train, test, logdir, FLAGS.keep_ckpts)
    train.stop()
    objax.util.multi_host_barrier()


if __name__ == '__main__':
    flags.DEFINE_enum('arch', 'wrn28-2', ARCHS, 'Model architecture.')
    flags.DEFINE_float('lr', 0.03, 'Learning rate.')
    flags.DEFINE_float('lr_decay', 0.25, 'Learning rate decay.')
    flags.DEFINE_float('wd', 0.001, 'Weight decay.')
    flags.DEFINE_integer('wu', 1, 'Iteration for phase3 (unlabeled weight loss for G).')
    flags.DEFINE_integer('batch', 64, 'Batch size')
    flags.DEFINE_integer('uratio', 3, 'Unlabeled batch size ratio')
    flags.DEFINE_integer('report_kimg', 64, 'Reporting period in kibi-images.')
    flags.DEFINE_integer('train_mimg', 8, 'Training duration in mega-images.')
    flags.DEFINE_integer('keep_ckpts', 5, 'Number of checkpoints to keep (0 for all).')
    flags.DEFINE_string('logdir', 'experiments', 'Directory where to save checkpoints and tensorboard data.')
    flags.DEFINE_string('dataset', 'domainnet32_infograph(10,seed=1)', 'Data to train on.')
    FLAGS.set_default('augment', 'CTA(sm,sm,probe=1)')
    FLAGS.set_default('para_augment', 8)
    app.run(main)
