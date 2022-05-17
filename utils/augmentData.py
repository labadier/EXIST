#%%

import pandas as pd, os, csv
from params import params

def dataAugmentation() -> None:
  data_frame = pd.read_csv(os.path.join(params.root, 'data/MAMI/training.csv'), sep='\t')
  with open(os.path.join(params.root, 'data/augmented.csv'), 'wt', newline='', encoding="utf-8") as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(params.columns)
    
    print("mami", len(data_frame))
    for row in data_frame.iterrows():
      spamwriter.writerow(['MAMI'] + ["en", row[1]['Text Transcription'].replace('\n', ' '), row[1]['misogynous'], row[1]['shaming'], 
      row[1]['stereotype'], row[1]['objectification'], row[1]['violence'], int(row[1]['misogynous'] == 1)])

  data_frame = pd.read_csv(os.path.join(params.root, 'data/HAHA/filtered.csv'))

  print("haha", len(data_frame))
  with open(os.path.join(params.root, 'data/augmented.csv'), 'at', newline='', encoding="utf-8") as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for row in data_frame.iterrows():
      spamwriter.writerow(['HAHA'] + ["es", row[1]['text'].replace('\n', ' '), 1] + [int(i == row[1]['humor_target']) for i in params.columns[4:] ])

  data_frame = pd.read_csv(os.path.join(params.root, 'data/HAHACKATHON/train.csv'))
  data_frame = data_frame[data_frame.apply(lambda x: x.astype(str).str.lower())['text'].str.contains('|'.join(params.sexistPhrase))][data_frame['offense_rating'] > 1.0]

  print("hahackathon", len(data_frame))
  with open(os.path.join(params.root, 'data/augmented.csv'), 'at', newline='', encoding="utf-8") as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for row in data_frame.iterrows():
      spamwriter.writerow(['HAHACKATHON'] + ["en", row[1]['text'].replace('\n', ' '), 1] + [-1]*5)


  data_frame = pd.read_csv(os.path.join(params.root, 'data/EXIST/training/EXIST2021_training.tsv'), sep='\t')
  with open(os.path.join(params.root, 'data/augmented.csv'), 'at', newline='', encoding="utf-8") as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for row in data_frame.iterrows():
      spamwriter.writerow(['EXIST'] + [row[1]['language'],  row[1]['text'].replace('\n', ' ') , int(row[1]['task1'] == "sexist")] + [int(i == row[1]['task2']) for i in params.columns_exist])

if __name__ == '__main__':
  dataAugmentation()
# %%
