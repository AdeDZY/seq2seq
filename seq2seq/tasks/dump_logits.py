# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Task where both the input and output sequence are plain text.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os

import numpy as np
from matplotlib import pyplot as plt

import tensorflow as tf
from tensorflow import gfile

from seq2seq.tasks.decode_text import _get_prediction_length
from seq2seq.tasks.prob_task import ProbTask, unbatch_dict


def _get_scores(predictions_dict):
  """Returns the logits, sliced by source and target length.
  """
  prediction_len = _get_prediction_length(predictions_dict)
  source_len = predictions_dict["features.source_len"]
  return predictions_dict["logits"][:prediction_len, :source_len]



class DumpLogits(ProbTask):
  """Defines inference for tasks where both the input and output sequences
  are plain text.

  Params:
    delimiter: Character by which tokens are delimited. Defaults to space.
    unk_replace: If true, enable unknown token replacement based on attention
      scores.
    unk_mapping: If `unk_replace` is true, this can be the path to a file
      defining a dictionary to improve UNK token replacement. Refer to the
      documentation for more details.
    dump_attention_dir: Save attention scores and plots to this directory.
    dump_attention_no_plot: If true, only save attention scores, not
      attention plots.
    dump_beams: Write beam search debugging information to this file.
  """

  def __init__(self, params):
    super(DumpLogits, self).__init__(params)
    self._logits_accum = []
    self._idx = 0

    if not self.params["output_dir"]:
      raise ValueError("Must specify output_dir for DumpLogits")

  @staticmethod
  def default_params():
    params = {}
    params.update({"output_dir": "", "dump_plots": True})
    return params

  def begin(self):
    super(DumpLogits, self).begin()
    gfile.MakeDirs(self.params["output_dir"])

  def before_run(self, _run_context):
    fetches = {}
    fetches["predicted_tokens"] = self._predictions["predicted_tokens"]
    fetches["features.source_len"] = self._predictions["features.source_len"]
    fetches["features.source_tokens"] = self._predictions[
        "features.source_tokens"]
    fetches["logits"] = self._predictions['logits']
    return tf.train.SessionRunArgs(fetches)

  def after_run(self, _run_context, run_values):
    fetches_batch = run_values.results
    for fetches in unbatch_dict(fetches_batch):
      # Convert to unicode
      fetches["predicted_tokens"] = np.char.decode(
          fetches["predicted_tokens"].astype("S"), "utf-8")
      fetches["features.source_tokens"] = np.char.decode(
          fetches["features.source_tokens"].astype("S"), "utf-8")

      self._logits_accum.append(_get_scores(fetches))

  def end(self, _session):
    scores_path = os.path.join(self.params["output_dir"],
                               "logits.npz")
    np.savez(scores_path, *self._logits_accum)
    tf.logging.info("Wrote %s", scores_path)
