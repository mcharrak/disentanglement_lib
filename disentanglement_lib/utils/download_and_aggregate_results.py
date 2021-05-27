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

"""Script to download all zip files containing each model's metrics and results.

To run this script, please change to project's home directory and run:
>> python disentanglement_lib/utils/download_and_aggregate_results.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from absl import app
from absl import flags

from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile


FLAGS = flags.FLAGS

# directory to store downloaded and unzipped files from url: https://storage.googleapis.com/disentanglement_lib/unsupervised_study_v1/<?>.zip
# <?> in the URL is in range [0,10799]
flags.DEFINE_string("output_dir", "output_directory/",
                    "Directory to save downloaded and unzipped files.")

flags.DEFINE_string("base_url", "https://storage.googleapis.com/disentanglement_lib/unsupervised_study_v1/",
                    "Base URL to zip files containing each model's results and metrics in .json format.")

flags.DEFINE_string("fname_json_agg_results", "aggregated_results.json",
                    "filename of aggregated .json file with all model results and metrics.")

def main(unused_argv):
    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)

    print('Downloading and unzipping zip files to {} ...'.format(FLAGS.output_dir))

# source: https://svaderia.github.io/articles/downloading-and-unzipping-a-zipfile/
    for i in range(10800):
        url_i = FLAGS.base_url + str(i) + ".zip"
        with urlopen(url_i) as zipresp:
            with ZipFile(BytesIO(zipresp.read())) as zfile:
                zfile.extractall(FLAGS.output_dir)

    print('Aggregating results as single .json file to {} ...'.format(FLAGS.output_dir))

    json_output = os.path.join(FLAGS.output_dir,FLAGS.fname_json_agg_results)

    os.system('dlib_aggregate_results --output_path={0} --result_file_pattern={1}/*/metrics/*/*/results/aggregate/evaluation.json'.format(json_output, FLAGS.output_dir))

    print('Downloading and aggregating results finished. Check {} directory for model folders and aggregated .json file.'.format(FLAGS.output_dir))

if __name__ == "__main__":
    app.run(main)