#%%
import translators as ts, pandas as pd
import os, csv
from params import params


for back_target in ['en', 'es']:
  with open(os.path.join(params.root, f'data/back_to_{back_target}.csv'), 'wt', newline='', encoding="utf-8") as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(params.columns)

for target_lang in ['en', 'es', 'fr', 'de']:

  print(f'Pivot Language {target_lang}: 0%', end="")

  perc = 0
  data_frame = pd.read_csv(os.path.join(params.root, 'data/augmented.csv'))

  with open(os.path.join(params.root, f'data/{target_lang}.csv'), 'wt', newline='', encoding="utf-8") as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(params.columns)
    
    for i, row in enumerate(data_frame.iterrows()):
      if (i*50.0)/len(data_frame) - perc > 1:
        perc = (i*50.0)/len(data_frame)
        print(f'\rPivot Language {target_lang}: {perc:.2f}%', end = "")

      new_row = row[1].copy()
      if new_row[1] != target_lang:
        try:
          new_row[2] = ts.google(new_row[2], from_language=new_row[1], to_language=target_lang)
        except:
          print(f'An exception occurred {new_row[2]}')
      spamwriter.writerow(new_row)

  data_frame = pd.read_csv(os.path.join(params.root,  f'data/{target_lang}.csv'))
  for back_target in ['en', 'es']:

    with open(os.path.join(params.root, f'data/back_to_{back_target}.csv'), 'at', newline='', encoding="utf-8") as csvfile:
      spamwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

      for row in data_frame.iterrows():
        if (i*50.0/len(data_frame)) - perc > 51.0:
          perc = 50.0 + (i*50.0/len(data_frame))
          print(f'\rPivot Language {target_lang}: {perc:.2f}%', end = "")

        new_row = row[1].copy()
        if back_target != target_lang:
          try:
            new_row[2] = ts.google(new_row[2], from_language=new_row[1], to_language=target_lang)
          except:
            print(f'An exception occurred: {new_row[2]}')
        spamwriter.writerow(new_row)
  print()

 # %%
