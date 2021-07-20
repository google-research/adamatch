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
import objax

from shared.data.fsl import DATASETS


def save(it, fn='/tmp/delme.png'):
    d = next(it)
    open(fn, 'wb').write(objax.util.image.to_png(d['image'].numpy().transpose((2, 0, 1))))
    return d['label'].numpy()


data = DATASETS()['domainnet64_infograph-0']()
it = iter(data.train.parse(1))
print(save(it))
# And repeat to overwrite delme.png with next image and so on.
# Also you can view images (and it automatically refreshes) using "eog delme.png"
print(save(it))
