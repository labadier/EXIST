#%%

import pandas as pd, os, csv

columns = ['campain', 'lang', 'tweet','sexism','ideological','stereotype','object','violence','mysogeny']
columns_exist = ['ideological-inequality', 'stereotyping-dominance', 'objectification', 'sexual-violence', 'misogyny-non-sexual-violence']
sexistPhrase = ['bitch', 'women', 'woman', 'femini', 'working mothers', 'office mom', 'man up',
                'lady boss', 'female ceo', 'mom']

root = '..'

data_frame = pd.read_csv(os.path.join(root, 'data/MAMI/training.csv'), sep='\t')
with open(os.path.join(root, 'data/augmented.csv'), 'wt', newline='', encoding="utf-8") as csvfile:
  spamwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
  spamwriter.writerow(columns)
  
  for row in data_frame.iterrows():
    spamwriter.writerow(['MAMI'] + ["en", row[1]['Text Transcription'], row[1]['misogynous'], row[1]['shaming'], 
    row[1]['stereotype'], row[1]['objectification'], row[1]['violence'], int(row[1]['misogynous'] == 1)])

data_frame = pd.read_csv(os.path.join(root, 'data/HAHA/filtered.csv'))
with open(os.path.join(root, 'data/augmented.csv'), 'at', newline='', encoding="utf-8") as csvfile:
  spamwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
  for row in data_frame.iterrows():
    spamwriter.writerow(['HAHA'] + ["es", row[1]['text'] , 1] + [int(i == row[1]['humor_target']) for i in columns[4:]])

data_frame = pd.read_csv(os.path.join(root, 'data/HAHACKATHON/train.csv'))
data_frame = data_frame[data_frame.apply(lambda x: x.astype(str).str.lower())['text'].str.contains('|'.join(sexistPhrase))][data_frame['offense_rating'] > 1.0]

with open(os.path.join(root, 'data/augmented.csv'), 'at', newline='', encoding="utf-8") as csvfile:
  spamwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
  for row in data_frame.iterrows():
    spamwriter.writerow(['HAHACKATHON'] + ["en", row[1]['text'] , 1] + [-1]*5)


data_frame = pd.read_csv(os.path.join(root, 'data/EXIST/training/EXIST2021_training.tsv'), sep='\t')
with open(os.path.join(root, 'data/augmented.csv'), 'at', newline='', encoding="utf-8") as csvfile:
  spamwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
  for row in data_frame.iterrows():
    spamwriter.writerow(['EXIST'] + [row[1]['language'],  row[1]['text'] , int(row[1]['task1'] == "sexist")] + [int(i == row[1]['task2']) for i in columns_exist])

# %%
