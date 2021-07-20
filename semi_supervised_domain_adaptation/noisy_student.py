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
"""NoisyStudent.
"""
import os
import sys
from typing import Callable

import jax
import numpy as np
import objax
from absl import app
from absl import flags
from absl.flags import FLAGS
from objax.typing import JaxArray
from tqdm import tqdm

from semi_supervised_domain_adaptation.lib.data import MixData, CTAData
from semi_supervised_domain_adaptation.lib.train import TrainableSSDAModule
from shared.data.fsl import DATASETS as FSL_DATASETS
from shared.data.ssl import DATASETS as SSL_DATASETS, DataSetSSL
from shared.data.augment.augment import AugmentPoolBase
from shared.train import ScheduleCos
from shared.util import setup_tf, MyParallel
from shared.zoo.models import network, ARCHS


class NoisyStudent(TrainableSSDAModule):
    def __init__(self, nclass: int, model: Callable, **kwargs):
        super().__init__(nclass, kwargs)
        self.model: objax.Module = model(colors=3, nclass=nclass, **kwargs)
        self.model_ema = objax.optimizer.ExponentialMovingAverageModule(self.model, momentum=0.999)
        if FLAGS.arch.endswith('pretrain'):
            # Initialize weights of EMA with pretrained model's weights.
            self.model_ema.ema.momentum = 0
            self.model_ema.update_ema()
            self.model_ema.ema.momentum = 0.999
        train_vars = self.model.vars()
        self.opt = objax.optimizer.Momentum(train_vars)
        self.lr = ScheduleCos(self.params.lr, self.params.lr_decay)

        @objax.Function.with_vars(self.model_ema.vars())
        def eval_op(x: JaxArray, domain: int) -> JaxArray:
            return objax.functional.softmax(self.model_ema(x, training=False, domain=domain))

        def loss_function(sx, sy):
            c, h, w = sx.shape[-3:]
            logit_sx = self.model(sx.reshape((-1, c, h, w)), training=True)
            logit_sx_weak, logit_sx_strong = logit_sx[::2], logit_sx[1::2]
            xe = 0.5 * (objax.functional.loss.cross_entropy_logits(logit_sx_weak, sy).mean() +
                        objax.functional.loss.cross_entropy_logits(logit_sx_strong, sy).mean())
            wd = 0.5 * sum((v.value ** 2).sum() for k, v in train_vars.items() if k.endswith('.w'))

            loss = xe + self.params.wd * wd
            return loss, {'losses/xe': xe, 'losses/wd': wd}

        gv = objax.GradValues(loss_function, train_vars)

        @objax.Function.with_vars(self.vars())
        def train_op(step, sx, sy, tx, ty, tu, probe=None):
            y_probe = eval_op(probe, 1) if probe is not None else None
            p = step / (FLAGS.train_mimg << 20)
            lr = self.lr(p)
            g, v = gv(sx, sy)
            self.opt(lr, objax.functional.parallel.pmean(g))
            self.model_ema.update_ema()
            return objax.functional.parallel.pmean({'monitors/lr': lr, **v[1]}), y_probe

        self.train_op = MyParallel(train_op, reduce=lambda x: x)
        self.eval_op = MyParallel(eval_op, static_argnums=(1,))
        # jit eval op used only for saving prediction files, avoids batch padding
        self.eval_op_jit = objax.Jit(eval_op, static_argnums=(1,))


class PseudoLabeler(AugmentPoolBase):
    def __init__(self, data: AugmentPoolBase, pseudo_label_file: str, threshold: float, uratio: float = 1):
        with open(pseudo_label_file, 'rb') as f:
            self.pseudo_label = np.load(f)
        self.has_pseudo_label = self.pseudo_label.max(axis=1) > threshold
        self.data = data
        self.uratio = uratio

    def stop(self):
        self.data.stop()

    def __iter__(self) -> dict:
        batch = []
        batch_size = None
        for d in self.data:
            batch_size = batch_size or d['sx']['image'].shape[0]
            for p, index in enumerate(d['sx']['index']):
                batch.append(dict(index=index, label=d['sx']['label'][p], image=d['sx']['image'][p]))
            for p, index in enumerate(d['tx']['index']):
                batch.append(dict(index=index, label=d['tx']['label'][p], image=d['tx']['image'][p]))
            for p, index in enumerate(d['tu']['index']):
                if self.has_pseudo_label[index]:
                    batch.append(dict(index=index, label=self.pseudo_label[index], image=d['tu']['image'][p]))
            np.random.shuffle(batch)
            while len(batch) >= batch_size:
                current_batch, batch = batch[:batch_size], batch[batch_size:]
                d['sx'] = dict(image=np.stack([x['image'] for x in current_batch]),
                               label=np.stack([x['label'] for x in current_batch]),
                               index=np.stack([x['index'] for x in current_batch]))
                yield d


def main(argv):
    del argv
    assert FLAGS.id, f'You must specify a --id for the run'
    print('JAX host: %d / %d' % (jax.host_id(), jax.host_count()))
    print('JAX devices:\n%s' % '\n'.join(str(d) for d in jax.devices()), flush=True)
    setup_tf()
    source = FSL_DATASETS()[f'{FLAGS.dataset}_{FLAGS.source}-0']()
    target_name, target_samples_per_class, target_seed = DataSetSSL.parse_name(f'{FLAGS.dataset}_{FLAGS.target}')
    target_labeled = SSL_DATASETS()[target_name](target_samples_per_class, target_seed)
    target_unlabeled = FSL_DATASETS()[f'{target_name}-0']()
    testsets = [target_unlabeled.test, source.test]  # Ordered by domain (unlabeled always first)
    module = NoisyStudent(source.nclass, network(FLAGS.arch),
                          lr=FLAGS.lr,
                          lr_decay=FLAGS.lr_decay,
                          wd=FLAGS.wd,
                          arch=FLAGS.arch,
                          batch=FLAGS.batch,
                          uratio=FLAGS.uratio)
    logdir = f'SSDA/{FLAGS.dataset}/{FLAGS.source}/{FLAGS.target}/{FLAGS.augment}/{module.__class__.__name__}/'
    logdir += '_'.join(sorted('%s%s' % k for k in module.params.items()))
    if FLAGS.teacher_id != '':
      pseudo_label_logdir = os.path.join(FLAGS.logdir, logdir, FLAGS.teacher_id)
      pseudo_label_file = os.path.join(pseudo_label_logdir, 'predictions.npy')
      print('Using pseudo label file ', pseudo_label_file)
    else:
      pseudo_label_file = FLAGS.pseudo_label_file
    logdir = os.path.join(FLAGS.logdir, logdir, FLAGS.id)
    prediction_file = os.path.join(logdir, 'predictions.npy')
    assert not os.path.exists(prediction_file), f'The prediction file "{prediction_file}" already exists, ' \
                                                f'remove it if you want to overwrite it.'

    test = {}
    for domain, testset in enumerate(testsets):
        test.update((k, v.parse().batch(FLAGS.batch).nchw().map(lambda d: {**d, 'domain': domain}).prefetch(16))
                    for k, v in testset.items())
    if FLAGS.augment.startswith('('):
        train = MixData(source.train, target_labeled.train, target_unlabeled.train, source.nclass, FLAGS.batch,
                        FLAGS.uratio)
    elif FLAGS.augment.startswith('CTA('):
        train = CTAData(source.train, target_labeled.train, target_unlabeled.train, source.nclass, FLAGS.batch,
                        FLAGS.uratio)
    else:
        raise ValueError(f'Augment flag value {FLAGS.augment} not supported.')
    if pseudo_label_file != '':
        train = PseudoLabeler(train, pseudo_label_file, FLAGS.pseudo_label_th, FLAGS.uratio)
    module.train(FLAGS.train_mimg << 10, FLAGS.report_kimg, train, test, logdir, FLAGS.keep_ckpts)
    predictions = []
    for batch in tqdm(target_unlabeled.train.parse().batch(FLAGS.batch).nchw(),
                        desc=f'Evaluating unlabeled {FLAGS.target}', leave=False):
        predictions.append(module.eval_op_jit(batch['image']._numpy(), 0))
    predictions = np.concatenate(predictions)
    with open(prediction_file, 'wb') as f:
        np.save(f, predictions)
    print('Saved target predictions to', prediction_file)
    train.stop()


if __name__ == '__main__':
    flags.DEFINE_enum('arch', 'wrn28-2', ARCHS, 'Model architecture.')
    flags.DEFINE_float('lr', 0.03, 'Learning rate.')
    flags.DEFINE_float('lr_decay', 0.25, 'Learning rate decay.')
    flags.DEFINE_float('pseudo_label_th', 0.9, 'Pseudo-label threshold.')
    flags.DEFINE_float('wd', 0.001, 'Weight decay.')
    flags.DEFINE_integer('batch', 64, 'Batch size')
    flags.DEFINE_integer('uratio', 3, 'Unlabeled batch size ratio')
    flags.DEFINE_integer('report_kimg', 64, 'Reporting period in kibi-images.')
    flags.DEFINE_integer('train_mimg', 8, 'Training duration in mega-images.')
    flags.DEFINE_integer('keep_ckpts', 5, 'Number of checkpoints to keep (0 for all).')
    flags.DEFINE_string('id', '', 'Id of the experiments.')
    flags.DEFINE_string('pseudo_label_file', '', 'Prediction file to read.')
    flags.DEFINE_string('teacher_id', '', 'Exp id of teacher. Overrides pseudo_label_file if set.')
    flags.DEFINE_string('logdir', 'experiments', 'Directory where to save checkpoints and tensorboard data.')
    flags.DEFINE_string('dataset', 'domainnet32', 'Source data to train on.')
    flags.DEFINE_string('source', 'clipart', 'Source data to train on.')
    flags.DEFINE_string('target', 'infograph(10,seed=1)', 'Target data to train on.')
    FLAGS.set_default('augment', 'CTA(sm,sm,probe=1)')
    FLAGS.set_default('para_augment', 8)
    app.run(main)
