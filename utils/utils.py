import pandas as pd, numpy as np, os
from utils.params import params

def load_data(filename):

  data = pd.read_csv(filename, dtype=str)
  text = data['tweet'].to_numpy()[:16]
    
  if len(data.keys()) < 4:
    return text
  labels = data[params.columns[3:]].astype(int).to_numpy()[:16]
  return text, labels
