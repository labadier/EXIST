import pandas as pd, numpy as np, os
from utils.params import params

def load_data(filename):

  data = pd.read_csv(filename, dtype=str)
  text = data['tweet'].to_numpy()
    
  if len(data.keys()) < 4:
    return text
  labels = data[params.columns[3:]].to_numpy()
  return text, labels
