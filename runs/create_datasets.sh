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
CUDA_VISIBLE_DEVICES= python scripts/create_datasets.py office3132_amazon &
CUDA_VISIBLE_DEVICES= python scripts/create_datasets.py cifar10 &
CUDA_VISIBLE_DEVICES= python scripts/create_datasets.py cifar100 &
CUDA_VISIBLE_DEVICES= python scripts/create_datasets.py mnist &
CUDA_VISIBLE_DEVICES= python scripts/create_datasets.py svhn &
CUDA_VISIBLE_DEVICES= python scripts/create_datasets.py domainnet32_clipart &
CUDA_VISIBLE_DEVICES= python scripts/create_datasets.py domainnet32_infograph &
CUDA_VISIBLE_DEVICES= python scripts/create_datasets.py domainnet32_quickdraw &
CUDA_VISIBLE_DEVICES= python scripts/create_datasets.py domainnet32_painting &
CUDA_VISIBLE_DEVICES= python scripts/create_datasets.py domainnet32_real &
CUDA_VISIBLE_DEVICES= python scripts/create_datasets.py domainnet32_sketch &
wait
CUDA_VISIBLE_DEVICES= python scripts/create_datasets.py office3132_dslr &
CUDA_VISIBLE_DEVICES= python scripts/create_datasets.py office3132_webcam &
CUDA_VISIBLE_DEVICES= python scripts/create_datasets.py office3164_amazon &
CUDA_VISIBLE_DEVICES= python scripts/create_datasets.py office3164_dslr &
CUDA_VISIBLE_DEVICES= python scripts/create_datasets.py office3164_webcam &
CUDA_VISIBLE_DEVICES= python scripts/create_datasets.py office31128_amazon &
CUDA_VISIBLE_DEVICES= python scripts/create_datasets.py office31128_dslr &
CUDA_VISIBLE_DEVICES= python scripts/create_datasets.py office31128_webcam &
CUDA_VISIBLE_DEVICES= python scripts/create_datasets.py office31224_amazon &
CUDA_VISIBLE_DEVICES= python scripts/create_datasets.py office31224_dslr &
CUDA_VISIBLE_DEVICES= python scripts/create_datasets.py office31224_webcam &
CUDA_VISIBLE_DEVICES= python scripts/create_datasets.py domainnet64_clipart &
CUDA_VISIBLE_DEVICES= python scripts/create_datasets.py domainnet64_infograph &
CUDA_VISIBLE_DEVICES= python scripts/create_datasets.py domainnet64_quickdraw &
CUDA_VISIBLE_DEVICES= python scripts/create_datasets.py domainnet64_painting &
CUDA_VISIBLE_DEVICES= python scripts/create_datasets.py domainnet64_real &
CUDA_VISIBLE_DEVICES= python scripts/create_datasets.py domainnet64_sketch &
CUDA_VISIBLE_DEVICES= python scripts/create_datasets.py domainnet128_clipart &
CUDA_VISIBLE_DEVICES= python scripts/create_datasets.py domainnet128_infograph &
CUDA_VISIBLE_DEVICES= python scripts/create_datasets.py domainnet128_quickdraw &
CUDA_VISIBLE_DEVICES= python scripts/create_datasets.py domainnet128_painting &
CUDA_VISIBLE_DEVICES= python scripts/create_datasets.py domainnet128_real &
CUDA_VISIBLE_DEVICES= python scripts/create_datasets.py domainnet128_sketch &
CUDA_VISIBLE_DEVICES= python scripts/create_datasets.py domainnet224_clipart &
CUDA_VISIBLE_DEVICES= python scripts/create_datasets.py domainnet224_infograph &
CUDA_VISIBLE_DEVICES= python scripts/create_datasets.py domainnet224_quickdraw &
CUDA_VISIBLE_DEVICES= python scripts/create_datasets.py domainnet224_painting &
CUDA_VISIBLE_DEVICES= python scripts/create_datasets.py domainnet224_real &
CUDA_VISIBLE_DEVICES= python scripts/create_datasets.py domainnet224_sketch &
CUDA_VISIBLE_DEVICES= python scripts/create_datasets.py digitfive32_mnist &
CUDA_VISIBLE_DEVICES= python scripts/create_datasets.py digitfive32_mnistm &
CUDA_VISIBLE_DEVICES= python scripts/create_datasets.py digitfive32_svhn &
CUDA_VISIBLE_DEVICES= python scripts/create_datasets.py digitfive32_syndigit &
CUDA_VISIBLE_DEVICES= python scripts/create_datasets.py digitfive32_usps &
wait
