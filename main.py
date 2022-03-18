#%%
from utils.params import params
from utils.utils import load_data

text, label  = load_data('data/back_to_en.csv')
# %%

with open('test_text.out', 'w') as file:
  for i in label:
    file.write(f'{i}\n') 
# %%
