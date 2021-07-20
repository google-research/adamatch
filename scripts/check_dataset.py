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
from absl import app
from absl import flags

from shared.data.fsl import DATASETS

FLAGS = flags.FLAGS


def main(argv):
    del argv  # Unused.
    dataset = DATASETS()[FLAGS.dataset]()
    data = dataset.train.parse(1)

    # Iterate through data
    for it in data:
        print(it['image'][0])
        break


if __name__ == '__main__':
    flags.DEFINE_string('dataset', 'domainnet32_infograph-0', 'Data to check.')
    app.run(main)
