from typing import Union, Tuple, Dict, Any, Optional
import os
import json
from collections import OrderedDict
import torch

class PretrainedConfig(object):
  model_type: str = ""
  is_composition: bool = False

  def __init__(self, **kwargs):
    # Attributes with defaults
    self.return_dict = kwargs.pop("return_dict", True)
    self.output_hidden_states = kwargs.pop("output_hidden_states", False)
    self.output_attentions = kwargs.pop("output_attentions", False)
    self.torchscript = kwargs.pop("torchscript", False)  # Only used by PyTorch models
    self.use_bfloat16 = kwargs.pop("use_bfloat16", False)
    self.pruned_heads = kwargs.pop("pruned_heads", {})
    self.tie_word_embeddings = kwargs.pop(
      "tie_word_embeddings", True
    )  # Whether input and output word embeddings should be tied for all MLM, LM and Seq2Seq models.

    # Is decoder is used in encoder-decoder models to differentiate encoder from decoder
    self.is_encoder_decoder = kwargs.pop("is_encoder_decoder", False)
    self.is_decoder = kwargs.pop("is_decoder", False)
    self.add_cross_attention = kwargs.pop("add_cross_attention", False)
    self.tie_encoder_decoder = kwargs.pop("tie_encoder_decoder", False)

    # Parameters for sequence generation
    self.max_length = kwargs.pop("max_length", 20)
    self.min_length = kwargs.pop("min_length", 0)
    self.do_sample = kwargs.pop("do_sample", False)
    self.early_stopping = kwargs.pop("early_stopping", False)
    self.num_beams = kwargs.pop("num_beams", 1)
    self.num_beam_groups = kwargs.pop("num_beam_groups", 1)
    self.diversity_penalty = kwargs.pop("diversity_penalty", 0.0)
    self.temperature = kwargs.pop("temperature", 1.0)
    self.top_k = kwargs.pop("top_k", 50)
    self.top_p = kwargs.pop("top_p", 1.0)
    self.repetition_penalty = kwargs.pop("repetition_penalty", 1.0)
    self.length_penalty = kwargs.pop("length_penalty", 1.0)
    self.no_repeat_ngram_size = kwargs.pop("no_repeat_ngram_size", 0)
    self.encoder_no_repeat_ngram_size = kwargs.pop("encoder_no_repeat_ngram_size", 0)
    self.bad_words_ids = kwargs.pop("bad_words_ids", None)
    self.num_return_sequences = kwargs.pop("num_return_sequences", 1)
    self.chunk_size_feed_forward = kwargs.pop("chunk_size_feed_forward", 0)
    self.output_scores = kwargs.pop("output_scores", False)
    self.return_dict_in_generate = kwargs.pop("return_dict_in_generate", False)
    self.forced_bos_token_id = kwargs.pop("forced_bos_token_id", None)
    self.forced_eos_token_id = kwargs.pop("forced_eos_token_id", None)

    # Fine-tuning task arguments
    self.architectures = kwargs.pop("architectures", None)
    self.finetuning_task = kwargs.pop("finetuning_task", None)
    self.id2label = kwargs.pop("id2label", None)
    self.label2id = kwargs.pop("label2id", None)
    if self.id2label is not None:
      kwargs.pop("num_labels", None)
      self.id2label = dict((int(key), value) for key, value in self.id2label.items())
      # Keys are always strings in JSON so convert ids to int here.
    else:
      self.num_labels = kwargs.pop("num_labels", 2)

    # Tokenizer arguments
    self.tokenizer_class = kwargs.pop("tokenizer_class", None)
    self.prefix = kwargs.pop("prefix", None)
    self.bos_token_id = kwargs.pop("bos_token_id", None)
    self.pad_token_id = kwargs.pop("pad_token_id", None)
    self.eos_token_id = kwargs.pop("eos_token_id", None)
    self.sep_token_id = kwargs.pop("sep_token_id", None)

    self.decoder_start_token_id = kwargs.pop("decoder_start_token_id", None)

    # task specific arguments
    self.task_specific_params = kwargs.pop("task_specific_params", None)

    # TPU arguments
    self.xla_device = kwargs.pop("xla_device", None)

    # Name or path to the pretrained checkpoint
    self._name_or_path = str(kwargs.pop("name_or_path", ""))

    # Drop the transformers version info
    kwargs.pop("transformers_version", None)

    # Additional attributes without default values
    for key, value in kwargs.items():
      try:
        setattr(self, key, value)
      except AttributeError as err:
        raise err

  @classmethod
  def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
    config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)
    return cls.from_dict(config_dict, **kwargs)

  @classmethod
  def _dict_from_json_file(cls, json_file: Union[str, os.PathLike]):
    with open(json_file, "r", encoding="utf-8") as reader:
      text = reader.read()
    return json.loads(text)

  @classmethod
  def from_dict(cls, config_dict: Dict[str, Any], **kwargs) -> "PretrainedConfig":
    return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)

    config = cls(**config_dict)

    if hasattr(config, "pruned_heads"):
      config.pruned_heads = dict((int(key), value) for key, value in config.pruned_heads.items())

    # Update config with kwargs if needed
    to_remove = []
    for key, value in kwargs.items():
      if hasattr(config, key):
        setattr(config, key, value)
        to_remove.append(key)
    for key in to_remove:
      kwargs.pop(key, None)

    if return_unused_kwargs:
      return config, kwargs
    else:
      return config

class LlamaConfig(PretrainedConfig):
  model_type = "llama"
  def __init__(
    self,
    vocab_size: int = 32000,
    dim: int = 512,
    dropout: int = 0.0,
    n_layers: int = 8,
    n_heads: int = 8,
    n_kv_heads: Optional[int] = 8,
    max_seq_len: int = 1024,
    layer_norm_eps: float = 1e-5,
    multiple_of: int = 32,
    hidden_dim: Optional[int] = None,
    position_embedding_type: str = "rotary",
    use_cache: bool = True,
    **kwargs
  ):
    super().__init__(**kwargs)

    self.vocab_size = vocab_size
    self.dim = dim
    self.dropout = dropout
    self.n_layers = n_layers
    self.n_heads = n_heads
    self.max_seq_len = max_seq_len
    self.n_kv_heads = n_kv_heads
    self.layer_norm_eps = layer_norm_eps
    self.multiple_of = multiple_of
    self.hidden_dim = hidden_dim
    self.position_embedding_type = position_embedding_type
    self.use_cache = use_cache