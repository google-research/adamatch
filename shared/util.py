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
import threading

import numpy as np
import objax
import tensorflow as tf
from objax.typing import JaxArray


class StoppableThread(threading.Thread):
    def __init__(self, *args, **kwargs):
        super(StoppableThread, self).__init__(*args, **kwargs)
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()


class MyParallel(objax.Parallel):
    def device_reshape(self, x: JaxArray) -> JaxArray:
        """Utility to reshape an input array in order to broadcast to multiple devices."""
        assert hasattr(x, 'ndim'), f'Expected JaxArray, got {type(x)}. If you are trying to pass a scalar to ' \
                                   f'parallel, first convert it to a JaxArray, for example np.float(0.5)'
        if x.ndim == 0:
            return np.broadcast_to(x, [self.ndevices])  # Use np instead of jnp, 2x faster.
        assert x.shape[0] % self.ndevices == 0, f'Must be able to equally divide batch {x.shape} among ' \
                                                f'{self.ndevices} devices, but does not go equally.'
        return x.reshape((self.ndevices, x.shape[0] // self.ndevices) + x.shape[1:])


def setup_tf():
    tf.config.experimental.set_visible_devices([], "GPU")
