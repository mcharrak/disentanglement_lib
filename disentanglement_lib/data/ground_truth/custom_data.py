# coding=utf-8
# Copyright 2018 The DisentanglementLib Authors.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Custom data used for new datasets that are not supported by the library."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from disentanglement_lib.data.ground_truth import ground_truth_data
from disentanglement_lib.data.ground_truth import util
import numpy as np
from six.moves import range
from tensorflow.compat.v1 import gfile

CUSTOMDATA_PATH = os.path.join(
    os.environ.get("DISENTANGLEMENT_LIB_DATA", "."), "custom_data",
    "custom_data.npz")

class CustomData(ground_truth_data.GroundTruthData):
  """Custom image data set.

  The ground-truth factors of variation are (in the default setting):
  0 - name_factor_0 (N_0 different values)
  1 - name_factor_1 (N_1 different values)
  2 - name_factor_2 (N_2 different values)
  3 - name_factor_3 (N_3 different values)
  4 - name_factor_4 (N_4 different values)
  :
  :
  n - name_factor_n (N_n different values)  
  """

  def __init__(self):
    with gfile.Open(CUSTOMDATA_PATH, "rb") as f:
      # load data
      data = np.load(file=f, allow_pickle=True)
    self.images = data["imgs"]
    self.data_shape = list(self.images.shape[1:]) # first dimension [dim0] is dataset size; as list [] because gaussian_encoder_model.py requires it
    self.factor_sizes = data["factor_sizes"]
    self.latent_factor_indices = list(range(len(self.factor_sizes)))
    self.factor_bases = np.prod(self.factor_sizes) / np.cumprod(self.factor_sizes)
    self.state_space = util.SplitDiscreteStateSpace(self.factor_sizes, self.latent_factor_indices)
    
  @property
  def num_factors(self):
    return self.state_space.num_latent_factors

  @property
  def factors_num_values(self):
    return [self.full_factor_sizes[i] for i in self.latent_factor_indices]

  @property
  def observation_shape(self):
    return self.data_shape

  def sample_factors(self, num, random_state):
    """Sample a batch of factors Y."""
    return self.state_space.sample_latent_factors(num, random_state)

  def sample_observations_from_factors(self, factors, random_state):
    """Sample a batch of observations X given a batch of factors Y."""        
    all_factors = self.state_space.sample_all_factors(factors, random_state)
    indices = np.array(np.dot(all_factors, self.factor_bases), dtype=np.int64)
    return self.images[indices]

  def _sample_factor(self, i, num, random_state):
    return random_state.randint(self.factor_sizes[i], size=num)