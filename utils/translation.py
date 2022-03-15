#%%
import translators as ts, pandas as pd
import os, csv

root = '..'
columns = ['campain', 'lang', 'tweet','sexism','ideological','stereotype','object','violence','mysogeny']

data_frame = pd.read_csv(os.path.join(root, 'data/augmented.csv'))

for target_lang in ['en', 'es', 'fr', 'de']:

  print(f'Pivot Language {target_lang}')

  perc = 0
  with open(os.path.join(root, f'data/{target_lang}.csv'), 'wt', newline='', encoding="utf-8") as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(columns)
    
    for i, row in enumerate(data_frame.iterrows()):
      if perc - (i*1.0/2.0*len(data_frame)) > 0.01:
        perc = (i*1.0/2.0*len(data_frame))
        print(f'\rPivot Language {target_lang}: {perc*100.0:.2f}%', end = "")

      new_row = row[1].copy()
      if new_row[1] != target_lang:
        new_row[2] = ts.google(new_row[2], from_language=new_row[1], to_language=target_lang)
      spamwriter.writerow(new_row)

  for back_target in ['en', 'es']:
    if target_lang == back_target:
      continue
    
    with open(os.path.join(root, f'data/back_to_{back_target}.csv'), 'wt', newline='', encoding="utf-8") as csvfile:
      spamwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
      spamwriter.writerow(columns)
      
      for row in data_frame.iterrows():
        if perc - 50.0 + (i*1.0/2.0*len(data_frame)) > 0.01:
          perc = 50.0 + (i*1.0/2.0*len(data_frame))
          print(f'\rPivot Language {target_lang}: {perc*100.0:.2f}%', end = "")

        new_row = row[1].copy()
        new_row[2] = ts.google(new_row[2], from_language=target_lang, to_language=back_target)
        spamwriter.writerow(new_row)
  print()

# %%
