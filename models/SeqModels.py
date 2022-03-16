import torch, os
import numpy as np
from transformers import AutoModel, AutoTokenizer
from utils.params import params


def HugginFaceLoad(language, weigths_source):

  prefix = 'data' if weigths_source == 'offline' else ''
  model = AutoModel.from_pretrained(os.path.join(prefix , params.langaugeModel[language]))
  tokenizer = AutoTokenizer.from_pretrained(os.path.join(prefix , params.langaugeModel[language]), do_lower_case=True, TOKENIZERS_PARALLELISM=True)

  return model, tokenizer
  
    