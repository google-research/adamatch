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
import multiprocessing
import os

import numpy as np
from absl import flags

SHARED_MEMORY_SIZE = int(os.environ.get('SHARED_MEMORY_SIZE', 1 << 30))
SHARED_MEMORY = multiprocessing.Array('f', SHARED_MEMORY_SIZE, lock=False)
SHARED_MEMORY_NP = np.ctypeslib.as_array(SHARED_MEMORY)

flags.DEFINE_string('augment', 'MixUp(sm,smc)',
                    'Dataset augmentation method:\n'
                    '  Augmentations primitives:\n'
                    '    x   = identity\n'
                    '    m   = mirror\n'
                    '    s   = shift\n'
                    '    sc  = shift+cutout\n'
                    '    sm  = shift+mirror\n'
                    '    smc = shift+mirror+cutout\n'
                    '  Pair augmentations:\n'
                    '    (weak,strong) = Standard augmentation\n'
                    '    CTA(weak,strong,depth:int=2,th:float=0.8,decay:float=0.99) = CTAugment\n')


class AugmentPoolBase:
    @staticmethod
    def round_mem(v: int, align: int = 16):
        return align * ((v + align - 1) // align)

    @staticmethod
    def get_np_arrays(shapes):
        offsets = np.cumsum([0] + [AugmentPoolBase.round_mem(np.prod(s)) for s in shapes[:-1]])
        return [SHARED_MEMORY_NP[o:o + np.prod(s)].reshape(s) for o, s in zip(offsets, shapes)]

    def check_mem_requirements(self):
        total = sum(self.round_mem(np.prod(v)) for v in self.shapes)
        assert total <= SHARED_MEMORY_SIZE, f'Too little shared memory, do: export SHARED_MEMORY_SIZE={total}'

    def stop(self):
        pass
