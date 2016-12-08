"""
Definition of a basic seq2seq model
"""

import seq2seq.training
import seq2seq.encoders
import seq2seq.decoders
from .model_base import Seq2SeqBase

class BasicSeq2Seq(Seq2SeqBase):
  """Basic Sequence2Sequence model with a unidirectional encoder and decoder. The last encoder
  state is used to initialize the decoder and thus both must share the same type of RNN cell.

  Args:
    source_vocab_info: An instance of `seq2seq.inputs.VocabInfo` for the source vocabulary
    target_vocab_info: An instance of `seq2seq.inputs.VocabInfo` for the target vocabulary
    params: A dictionary of hyperparameters
  """

  def __init__(self, source_vocab_info, target_vocab_info, params, name="basic_seq2seq"):
    super(BasicSeq2Seq, self).__init__(source_vocab_info, target_vocab_info, params, name)

  @staticmethod
  def default_params():
    params = Seq2SeqBase.default_params().copy()
    params.update({
      "rnn_cell.type": "LSTMCell",
      "rnn_cell.num_units": 128,
      "rnn_cell.dropout_input_keep_prob": 1.0,
      "rnn_cell.dropout_output_keep_prob": 1.0,
      "rnn_cell.num_layers": 1
    })
    return params

  def encode_decode(self, source, source_len, decoder_input_fn, target_len, labels=None):
    # Create Encoder
    encoder_cell = seq2seq.training.utils.get_rnn_cell(
      cell_type=self.params["rnn_cell.type"],
      num_units=self.params["rnn_cell.num_units"],
      num_layers=self.params["rnn_cell.num_layers"],
      dropout_input_keep_prob=self.params["rnn_cell.dropout_input_keep_prob"],
      dropout_output_keep_prob=self.params["rnn_cell.dropout_output_keep_prob"])
    encoder_fn = seq2seq.encoders.UnidirectionalRNNEncoder(encoder_cell)
    encoder_output = encoder_fn(source, source_len)

    # Create Decoder
    # Because we pass the state between encoder and decoder we must use the same cell
    decoder_cell = encoder_cell
    decoder_fn = seq2seq.decoders.BasicDecoder(
      cell=decoder_cell,
      vocab_size=self.target_vocab_info.total_size,
      max_decode_length=self.params["target.max_seq_len"])

    decoder_output, _, _ = decoder_fn(
      input_fn=decoder_input_fn,
      initial_state=encoder_output.final_state,
      sequence_length=target_len)

    return decoder_output