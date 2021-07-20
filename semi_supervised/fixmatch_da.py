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
from typing import Callable

import jax
import jax.numpy as jn
import objax
from absl import app, flags
from absl.flags import FLAGS
from objax.functional import stop_gradient
from objax.typing import JaxArray

from semi_supervised.lib.data import MixData, CTAData
from semi_supervised.lib.train import TrainableSSLModule
from shared.data.fsl import DATASETS as FSL_DATASETS
from shared.data.ssl import DATASETS as SSL_DATASETS, DataSetSSL
from shared.train import ScheduleCos
from shared.util import setup_tf, MyParallel
from shared.zoo.models import network, ARCHS


class FixMatchDA(TrainableSSLModule):
    def __init__(self, nclass: int, model: Callable, **kwargs):
        super().__init__(nclass, kwargs)
        self.model: objax.Module = model(colors=3, nclass=nclass, **kwargs)
        self.model_ema = objax.optimizer.ExponentialMovingAverageModule(self.model, momentum=0.999)
        if FLAGS.arch.endswith('pretrain'):
            # Initialize weights of EMA with pretrained model's weights.
            self.model_ema.ema.momentum = 0
            self.model_ema.update_ema()
            self.model_ema.ema.momentum = 0.999
        self.stats = objax.Module()
        self.stats.p_data = objax.nn.ExponentialMovingAverage((nclass,), init_value=1 / nclass)
        self.stats.p_model = objax.nn.MovingAverage((nclass,), buffer_size=128, init_value=1 / nclass)
        train_vars = self.model.vars() + self.stats.vars()
        self.opt = objax.optimizer.Momentum(train_vars)
        self.lr = ScheduleCos(self.params.lr, self.params.lr_decay)

        @objax.Function.with_vars(self.model_ema.vars())
        def eval_op(x: JaxArray) -> JaxArray:
            return objax.functional.softmax(self.model_ema(x, training=False))

        def loss_function(x, y, u):
            c, h, w = x.shape[-3:]
            xu = jn.concatenate((x, u)).reshape((-1, c, h, w))
            logit = self.model(xu, training=True)
            logit_x_weak = logit[:2 * x.shape[0]:2]
            logit_weak, logit_strong = logit[::2], logit[1::2]


            confidence_ratio = self.params.confidence

            pseudo_labels = objax.functional.softmax(logit_weak)
            p_data = self.stats.p_data(y.mean(0))
            p_model = self.stats.p_model(pseudo_labels.mean(0))
            pseudo_labels *= (1e-6 + p_data) / (1e-6 + p_model)
            pseudo_labels = stop_gradient(pseudo_labels / pseudo_labels.sum(1, keepdims=True))
            pseudo_mask = (pseudo_labels.max(axis=1) >= confidence_ratio).astype(pseudo_labels.dtype)

            xe = objax.functional.loss.cross_entropy_logits(logit_x_weak, y).mean()
            xeu = objax.functional.loss.cross_entropy_logits_sparse(logit_strong, pseudo_labels.argmax(axis=1))
            xeu = (xeu * pseudo_mask).mean()
            wd = 0.5 * sum((v.value ** 2).sum() for k, v in train_vars.items() if k.endswith('.w'))

            loss = xe + self.params.wu * xeu + self.params.wd * wd
            return loss, {'losses/xe': xe,
                          'losses/xeu': xeu,
                          'losses/wd': wd,
                          'monitors/confidence_ratio': confidence_ratio,
                          'monitors/mask': pseudo_mask.mean(),
                          'monitors/klmodel': objax.functional.divergence.kl(p_data, p_model)}

        gv = objax.GradValues(loss_function, train_vars)

        @objax.Function.with_vars(self.vars())
        def train_op(step, x, y, u, probe=None):
            y_probe = eval_op(probe) if probe is not None else None
            p = step / (FLAGS.train_mimg << 20)
            lr = self.lr(p)
            g, v = gv(x, y, u)
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
    dataset_name, samples_per_class, dataset_seed = DataSetSSL.parse_name(f'{FLAGS.dataset}')
    labeled = SSL_DATASETS()[dataset_name](samples_per_class, dataset_seed)
    unlabeled = FSL_DATASETS()[f'{dataset_name}-0']()
    testsets = [unlabeled.test]
    module = FixMatchDA(labeled.nclass, network(FLAGS.arch),
                        lr=FLAGS.lr,
                        lr_decay=FLAGS.lr_decay,
                        wd=FLAGS.wd,
                        arch=FLAGS.arch,
                        batch=FLAGS.batch,
                        wu=FLAGS.wu,
                        confidence=FLAGS.confidence,
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
    flags.DEFINE_float('confidence', 0.9, 'Confidence threshold.')
    flags.DEFINE_float('lr', 0.03, 'Learning rate.')
    flags.DEFINE_float('lr_decay', 0.25, 'Learning rate decay.')
    flags.DEFINE_float('wd', 0.001, 'Weight decay.')
    flags.DEFINE_float('wu', 1, 'Unlabeled loss weight.')
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
