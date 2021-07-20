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
from typing import Dict, Iterable, Optional

import jax.numpy as jn
import numpy as np
from objax.jaxboard import Summary
from tqdm import tqdm

from shared.train import TrainableModule


class TrainableDAModule(TrainableModule):
    def eval(self, summary: Summary, epoch: int, test: Dict[str, Iterable], valid: Optional[Iterable] = None):
        def get_accuracy(dataset: Iterable):
            accuracy, total, batch = 0, 0, None
            for data in tqdm(dataset, leave=False, desc='Evaluating'):
                x, y, domain = data['image']._numpy(), data['label']._numpy(), int(data['domain']._numpy())
                total += x.shape[0]
                batch = batch or x.shape[0]
                if x.shape[0] != batch:
                    # Pad the last batch if it's smaller than expected (must divide properly on GPUs).
                    x = np.concatenate([x] + [x[-1:]] * (batch - x.shape[0]))
                p = self.eval_op(x, domain)[:y.shape[0]]
                accuracy += (np.argmax(p, axis=1) == data['label'].numpy()).sum()
            return accuracy / total if total else 0

        test_accuracy = {key: get_accuracy(value) for key, value in test.items()}
        to_print = []
        for key, value in sorted(test_accuracy.items()):
            summary.scalar('accuracy/%s' % key, 100 * value)
            to_print.append('Acccuracy/%s %.2f' % (key, summary['accuracy/%s' % key]()))
        print('Epoch %-4d  Loss %.2f  %s' % (epoch + 1, summary['losses/xe'](), ' '.join(to_print)))

    def train_step(self, summary, data, step):
        kv, probe = self.train_op(step,
                                  data['sx']['image'], data['sx']['label'],
                                  data['tu']['image'], data.get('probe'))
        if 'probe_callback' in data:
            data['probe_callback'](jn.concatenate(probe))
        for k, v in kv.items():
            v = v[0]
            if jn.isnan(v):
                raise ValueError('NaN', k)
            summary.scalar(k, float(v))
