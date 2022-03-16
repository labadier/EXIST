
from logging import root
from charset_normalizer import models


class params:

  root = '..'
  models = {'fr': 'flaubert/flaubert_base_cased', 'en': 'vinai/bertweet-base',
            'es':'finiteautomata/beto-sentiment-analysis', 'de':'oliverguhr/german-sentiment-bert'}
  
  columns = ['campain', 'lang', 'tweet','sexism','ideological','stereotype','object','violence','mysogeny']

  columns_exist = ['ideological-inequality', 'stereotyping-dominance', 'objectification', 'sexual-violence', 'misogyny-non-sexual-violence']
  
  sexistPhrase = ['bitch', 'women', 'woman', 'femini', 'working mothers', 'office mom', 'man up',
                'lady boss', 'female ceo', 'mom']


