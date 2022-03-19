
class params:

  root = '..'
  models = {'fr': 'flaubert/flaubert_base_cased', 'en': 'vinai/bertweet-base',
            'es':'finiteautomata/beto-sentiment-analysis', 'de':'oliverguhr/german-sentiment-bert'}
  
  columns = ['campain', 'lang', 'tweet','sexism','ideological','stereotype','object','violence','mysogeny']

  columns_exist = ['ideological-inequality', 'stereotyping-dominance', 'objectification', 'sexual-violence', 'misogyny-non-sexual-violence']
  
  sexistPhrase = ['bitch', 'women', 'woman', 'femini', 'working mothers', 'office mom', 'man up',
                'lady boss', 'female ceo', 'mom']

  
  LR, DECAY = 1e-5,  2e-5
  SPLITS = 5
  IL = 64
  ML = 130
  BS = 64
  EPOCHES = 4
  MULTITASK = 'stl'
  PRET_MODE = 'offline'
  
class bcolors:
  HEADER = '\033[95m'
  OKBLUE = '\033[94m'
  OKCYAN = '\033[96m'
  OKGREEN = '\033[92m'
  WARNING = '\033[93m'
  FAIL = '\033[91m'
  ENDC = '\033[0m'
  BOLD = '\033[1m'
  UNDERLINE = '\033[4m'


