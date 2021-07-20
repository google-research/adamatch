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

from semi_supervised.lib.data import MixData, CTAData
from semi_supervised.lib.train import TrainableSSLModule
from shared.data.augment.augment import AugmentPoolBase
from shared.data.fsl import DATASETS as FSL_DATASETS
from shared.data.ssl import DATASETS as SSL_DATASETS, DataSetSSL
from shared.train import ScheduleCos
from shared.util import setup_tf, MyParallel
from shared.zoo.models import network, ARCHS


class NoisyStudent(TrainableSSLModule):
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
        def eval_op(x: JaxArray) -> JaxArray:
            return objax.functional.softmax(self.model_ema(x, training=False))

        def loss_function(x, y):
            c, h, w = x.shape[-3:]
            logit_x = self.model(x.reshape((-1, c, h, w)), training=True)
            logit_x_weak, logit_x_strong = logit_x[::2], logit_x[1::2]
            xe = 0.5 * (objax.functional.loss.cross_entropy_logits(logit_x_weak, y).mean() +
                        objax.functional.loss.cross_entropy_logits(logit_x_strong, y).mean())
            wd = 0.5 * sum((v.value ** 2).sum() for k, v in train_vars.items() if k.endswith('.w'))

            loss = xe + self.params.wd * wd
            return loss, {'losses/xe': xe, 'losses/wd': wd}

        gv = objax.GradValues(loss_function, train_vars)

        @objax.Function.with_vars(self.vars())
        def train_op(step, x, y, u, probe=None):
            y_probe = eval_op(probe) if probe is not None else None
            p = step / (FLAGS.train_mimg << 20)
            lr = self.lr(p)
            g, v = gv(x, y)
            self.opt(lr, objax.functional.parallel.pmean(g))
            self.model_ema.update_ema()
            return objax.functional.parallel.pmean({'monitors/lr': lr, **v[1]}), y_probe

        self.train_op = MyParallel(train_op, reduce=lambda x: x)
        self.eval_op = MyParallel(eval_op)
        # jit eval op used only for saving prediction files, avoids batch padding
        self.eval_op_jit = objax.Jit(eval_op)


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
            batch_size = batch_size or d['x']['image'].shape[0]
            for p, index in enumerate(d['x']['index']):
                batch.append(dict(index=index, label=d['x']['label'][p], image=d['x']['image'][p]))
            for p, index in enumerate(d['u']['index']):
                if self.has_pseudo_label[index]:
                    batch.append(dict(index=index, label=self.pseudo_label[index], image=d['u']['image'][p]))
            np.random.shuffle(batch)
            while len(batch) >= batch_size:
                current_batch, batch = batch[:batch_size], batch[batch_size:]
                d['x'] = dict(image=np.stack([x['image'] for x in current_batch]),
                               label=np.stack([x['label'] for x in current_batch]),
                               index=np.stack([x['index'] for x in current_batch]))
                yield d


def main(argv):
    del argv
    assert FLAGS.id, f'You must specify a --id for the run'
    print('JAX host: %d / %d' % (jax.host_id(), jax.host_count()))
    print('JAX devices:\n%s' % '\n'.join(str(d) for d in jax.devices()), flush=True)
    setup_tf()
    dataset_name, samples_per_class, dataset_seed = DataSetSSL.parse_name(f'{FLAGS.dataset}')
    labeled = SSL_DATASETS()[dataset_name](samples_per_class, dataset_seed)
    unlabeled = FSL_DATASETS()[f'{dataset_name}-0']()
    testsets = [unlabeled.test]
    module = NoisyStudent(labeled.nclass, network(FLAGS.arch),
                          lr=FLAGS.lr,
                          lr_decay=FLAGS.lr_decay,
                          wd=FLAGS.wd,
                          arch=FLAGS.arch,
                          batch=FLAGS.batch,
                          uratio=FLAGS.uratio)
    logdir = f'SSL/{FLAGS.dataset}/{FLAGS.augment}/{module.__class__.__name__}/'
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
        test.update((k, v.parse().batch(FLAGS.batch).nchw().map(
            lambda d: {**d, 'domain': domain}).prefetch(16))
            for k, v in testset.items())
    if FLAGS.augment.startswith('('):
        train = MixData(labeled.train, unlabeled.train, labeled.nclass, FLAGS.batch, FLAGS.uratio)
    elif FLAGS.augment.startswith('CTA('):
        train = CTAData(labeled.train, unlabeled.train, labeled.nclass, FLAGS.batch, FLAGS.uratio)
    else:
        raise ValueError(f'Augment flag value {FLAGS.augment} not supported.')
    if pseudo_label_file != '':
        train = PseudoLabeler(train, pseudo_label_file, FLAGS.pseudo_label_th, FLAGS.uratio)
    module.train(FLAGS.train_mimg << 10, FLAGS.report_kimg, train, test, logdir, FLAGS.keep_ckpts)
    predictions = []
    for batch in tqdm(unlabeled.train.parse().batch(FLAGS.batch).nchw(),
                        desc=f'Evaluating unlabeled data', leave=False):
        predictions.append(module.eval_op_jit(batch['image']._numpy()))
    predictions = np.concatenate(predictions)
    with open(prediction_file, 'wb') as f:
        np.save(f, predictions)
    print('Saved target predictions to', prediction_file)
    train.stop()
    objax.util.multi_host_barrier()


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
    flags.DEFINE_string('dataset', 'domainnet32_infograph(10,seed=1)', 'Data to train on.')
    FLAGS.set_default('augment', 'CTA(sm,sm,probe=1)')
    FLAGS.set_default('para_augment', 8)
    app.run(main)
