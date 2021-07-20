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
from objax.functional import stop_gradient
from objax.typing import JaxArray

from semi_supervised_domain_adaptation.lib.data import MixData, CTAData
from semi_supervised_domain_adaptation.lib.train import TrainableSSDAModule
from shared.data.fsl import DATASETS as FSL_DATASETS
from shared.data.ssl import DATASETS as SSL_DATASETS, DataSetSSL
from shared.train import ScheduleCos, ScheduleCosPhases
from shared.util import setup_tf, MyParallel
from shared.zoo.models import network, ARCHS


class AdaMatch(TrainableSSDAModule):
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
        self.stats.keygen = objax.random.DEFAULT_GENERATOR
        self.stats.p_labeled = objax.nn.ExponentialMovingAverage((nclass,), init_value=1 / nclass)
        self.stats.p_unlabeled = objax.nn.MovingAverage((nclass,), buffer_size=128, init_value=1 / nclass)
        train_vars = self.model.vars() + self.stats.vars()
        self.opt = objax.optimizer.Momentum(train_vars)
        self.wu = ScheduleCosPhases(1, [(0.5, 1), (1, self.params.wu)], start_value=0)
        self.lr = ScheduleCos(self.params.lr, self.params.lr_decay)

        @objax.Function.with_vars(self.model_ema.vars())
        def eval_op(x: JaxArray, domain: int) -> JaxArray:
            return objax.functional.softmax(self.model_ema(x, training=False, domain=domain))

        def loss_function(sx, sy, tx, ty, tu, progress):
            sx_domain = jn.ones(2 * sx.shape[0], dtype=jn.int32)
            tx_domain = jn.zeros(2 * tx.shape[0], dtype=jn.int32)
            tu_domain = jn.zeros(2 * tu.shape[0], dtype=jn.int32)
            saved_vars = self.model.vars().tensors()
            logit_bn = self.model(jn.concatenate((sx, tx)).reshape((-1, *sx.shape[2:])), training=True,
                                  domain=jn.concatenate((sx_domain, tx_domain)))
            logit_bn_sx, logit_bn_tx = jn.split(logit_bn, 2)
            self.model.vars().assign(saved_vars)

            xu = jn.concatenate((sx, tx, tu)).reshape((-1, *sx.shape[2:]))
            logit = self.model(xu, training=True, domain=jn.concatenate((sx_domain, tx_domain, tu_domain)))
            logit_sx, logit_tx, logit_tu = jn.split(logit, (2 * sx.shape[0], 2 * (sx.shape[0] + tx.shape[0])))
            logit_sx += (logit_bn_sx - logit_sx) * objax.random.uniform(logit_sx.shape)
            logit_tx += (logit_bn_tx - logit_tx) * objax.random.uniform(logit_tx.shape)
            logit_sx_weak, logit_sx_strong = logit_sx[::2], logit_sx[1::2]
            logit_tx_weak, logit_tx_strong = logit_tx[::2], logit_tx[1::2]
            logit_tu_weak, logit_tu_strong = logit_tu[::2], logit_tu[1::2]

            xe = 0.5 * (objax.functional.loss.cross_entropy_logits(logit_sx_weak, sy).mean() +
                        objax.functional.loss.cross_entropy_logits(logit_sx_strong, sy).mean())
            xet = 0.5 * (objax.functional.loss.cross_entropy_logits(logit_tx_weak, ty).mean() +
                         objax.functional.loss.cross_entropy_logits(logit_tx_strong, ty).mean())

            if self.params.use_cr:
                real_confidence = objax.functional.softmax(stop_gradient(logit_sx_weak))
                confidence_ratio = real_confidence.max(1).mean(0) * self.params.confidence
            else:
                confidence_ratio = self.params.confidence

            pseudo_labels = objax.functional.softmax(logit_tu_weak)
            p_labeled = self.stats.p_labeled(objax.functional.softmax(logit_sx_weak).mean(0))
            p_unlabeled = self.stats.p_unlabeled(pseudo_labels.mean(0))
            pseudo_labels *= (1e-6 + p_labeled) / (1e-6 + p_unlabeled)
            pseudo_labels = stop_gradient(pseudo_labels / pseudo_labels.sum(1, keepdims=True))
            pseudo_mask = (pseudo_labels.max(axis=1) >= confidence_ratio).astype(pseudo_labels.dtype)
            xeu = objax.functional.loss.cross_entropy_logits_sparse(logit_tu_strong, pseudo_labels.argmax(axis=1))
            xeu = (xeu * pseudo_mask).mean()

            wd = 0.5 * sum((v.value ** 2).sum() for k, v in train_vars.items() if k.endswith('.w'))

            loss = xe + xet + self.wu(progress) * xeu + self.params.wd * wd
            return loss, {'losses/xe': xe,
                          'losses/xet': xet,
                          'losses/xeu': xeu,
                          'losses/wd': wd,
                          'losses/hregbn': jn.square(logit_sx - logit_bn_sx).mean(),
                          'monitors/confidence_ratio': confidence_ratio,
                          'monitors/wu': self.wu(progress),
                          'monitors/mask': pseudo_mask.mean(),
                          'monitors/klmodel': objax.functional.divergence.kl(p_labeled, p_unlabeled)}

        gv = objax.GradValues(loss_function, train_vars)

        @objax.Function.with_vars(self.vars())
        def train_op(step, sx, sy, tx, ty, tu, probe=None):
            y_probe = eval_op(probe, 1) if probe is not None else None
            p = step / (FLAGS.train_mimg << 20)
            lr = self.lr(p)
            g, v = gv(sx, sy, tx, ty, tu, p)
            self.opt(lr, objax.functional.parallel.pmean(g))
            self.model_ema.update_ema()
            return objax.functional.parallel.pmean({'monitors/lr': lr, **v[1]}), y_probe

        self.train_op = MyParallel(train_op, reduce=lambda x: x)
        self.eval_op = MyParallel(eval_op, static_argnums=(1,))


def main(argv):
    del argv
    print('JAX host: %d / %d' % (jax.host_id(), jax.host_count()))
    print('JAX devices:\n%s' % '\n'.join(str(d) for d in jax.devices()), flush=True)
    setup_tf()
    source = FSL_DATASETS()[f'{FLAGS.dataset}_{FLAGS.source}-0']()
    target_name, target_samples_per_class, target_seed = DataSetSSL.parse_name(f'{FLAGS.dataset}_{FLAGS.target}')
    target_labeled = SSL_DATASETS()[target_name](target_samples_per_class, target_seed)
    target_unlabeled = FSL_DATASETS()[f'{target_name}-0']()
    testsets = [target_unlabeled.test, source.test]  # Ordered by domain (unlabeled always first)
    module = AdaMatch(source.nclass, network(FLAGS.arch),
                      lr=FLAGS.lr,
                      lr_decay=FLAGS.lr_decay,
                      wd=FLAGS.wd,
                      arch=FLAGS.arch,
                      batch=FLAGS.batch,
                      wu=FLAGS.wu,
                      confidence=FLAGS.confidence,
                      use_cr=FLAGS.use_cr,
                      uratio=FLAGS.uratio)
    logdir = f'SSDA/{FLAGS.dataset}/{FLAGS.source}/{FLAGS.target}/{FLAGS.augment}/{module.__class__.__name__}/'
    logdir += '_'.join(sorted('%s%s' % k for k in module.params.items()))
    logdir = os.path.join(FLAGS.logdir, logdir)
    test = {}
    for domain, testset in enumerate(testsets):
        test.update((f'{FLAGS.source}_to_{k}',
                     v.parse().batch(FLAGS.batch).nchw().map(lambda d: {**d, 'domain': domain}).prefetch(16))
                    for k, v in testset.items())
    if FLAGS.augment.startswith('('):
        train = MixData(source.train, target_labeled.train, target_unlabeled.train, source.nclass, FLAGS.batch,
                        FLAGS.uratio)
    elif FLAGS.augment.startswith('CTA('):
        train = CTAData(source.train, target_labeled.train, target_unlabeled.train, source.nclass, FLAGS.batch,
                        FLAGS.uratio)
    else:
        raise ValueError(f'Augment flag value {FLAGS.augment} not supported.')
    module.train(FLAGS.train_mimg << 10, FLAGS.report_kimg, train, test, logdir, FLAGS.keep_ckpts)
    train.stop()


if __name__ == '__main__':
    flags.DEFINE_enum('arch', 'wrn28-2', ARCHS, 'Model architecture.')
    flags.DEFINE_bool('use_cr', True, 'Make confidence threshold proportional to real data.')
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
    flags.DEFINE_string('dataset', 'domainnet32', 'Source data to train on.')
    flags.DEFINE_string('source', 'clipart', 'Source data to train on.')
    flags.DEFINE_string('target', 'infograph(10,seed=1)', 'Target data to train on.')
    FLAGS.set_default('augment', 'CTA(sm,sm,probe=1)')
    FLAGS.set_default('para_augment', 8)
    app.run(main)
