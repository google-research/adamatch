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

import objax

from shared.data.core import DATA_DIR, DataSet, record_parse

# filename = 'office3164_amazon-train.tfrecord'
filename = 'domainnet64_infograph-train.tfrecord'
size = 64, 64, 3

d = DataSet.from_files([os.path.join(DATA_DIR, filename)], size, parse_fn=record_parse)
d = d.parse(1).batch(256)
v = next(iter(d))
m = v['image'].numpy()
m = m.reshape((16, 16, *size))
m = m.transpose((4, 0, 2, 1, 3))
m = m.reshape((size[2], 16 * size[0], 16 * size[1]))
open('/tmp/delme.png', 'wb').write(objax.util.image.to_png(m))
