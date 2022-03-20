import argparse, sys, os, numpy as np, torch, random
from utils.params import params, bcolors
from utils.utils import load_data, plot_training
from models.SeqModels import train_model_CV

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

def check_params(args=None):
  
  parser = argparse.ArgumentParser(description='Language Model Encoder')

  parser.add_argument('-phase', metavar='phase', help='Phase')
  parser.add_argument('-output', metavar='output', default = params.OUTPUT, help='Output Path')
  parser.add_argument('-lr', metavar='lrate', default = params.LR , type=float, help='learning rate')
  parser.add_argument('-decay', metavar='decay', default = params.DECAY, type=float, help='learning rate decay')
  parser.add_argument('-splits', metavar='splits', default = params.SPLITS, type=int, help='spits cross validation')
  parser.add_argument('-ml', metavar='max_length', default = params.ML, type=int, help='Maximun Tweet Length')
  parser.add_argument('-wm', metavar='weigths_mode', default = params.PRET_MODE, help='Mode of pretraining weiths (online or offline)')
  parser.add_argument('-interm_layer', metavar='int_layer', default = params.IL, type=int, help='Intermediate layers neurons')
  parser.add_argument('-epoches', metavar='epoches', default=params.EPOCHES, type=int, help='Trainning Epoches')
  parser.add_argument('-bs', metavar='batch_size', default=params.BS, type=int, help='Batch Size')
  parser.add_argument('-l', metavar='lang', help='Language')
  parser.add_argument('-tf', metavar='train_file', help='Data Anotation Files for Training')
  parser.add_argument('-df', metavar='test_file', help='Data Anotation Files for Testing')
  parser.add_argument('-wp', metavar='weigths_path', default="logs", help='Saved Weights Path')
  parser.add_argument('-mtl', metavar='multitask', default=params.MULTITASK, help='Multitask Leatning')

  return parser.parse_args(args)


if __name__ == '__main__':


  parameters = check_params(sys.argv[1:])

  learning_rate, decay = parameters.lr,  parameters.decay
  splits = parameters.splits
  interm_layer_size = parameters.interm_layer
  max_length = parameters.ml
  batch_size = parameters.bs
  epoches = parameters.epoches
  phase = parameters.phase
  output = parameters.output
  model_mode = parameters.wm
  
  tf = parameters.tf
  df=parameters.df
  weights_path = parameters.wp
  lang = parameters.l
  multitask = (parameters.mtl == 'mtl')

  if phase == 'train':

    output = os.path.join(output, 'logs')

    if os.path.exists(output) == False:
      os.system(f'mkdir {output}')

    text, label  = load_data('data/back_to_en.csv')
    data = {'text':text, 'labels':label}
    
    history = train_model_CV(model_name=params.models[lang].split('/')[-1], lang=lang, data=data, splits=splits, epoches=epoches, 
                  batch_size=batch_size, max_length=max_length, interm_layer_size = interm_layer_size, 
                  lr = learning_rate,  decay=decay, output=output, multitask=multitask, model_mode=model_mode)
    
    print(f"{bcolors.OKCYAN}{bcolors.BOLD}Training Finished for {lang.upper()} Model{bcolors.ENDC}")
    plot_training(history[-1], 'lm_{lang}', output, 'acc')
    exit(0)

  if phase == 'eval':
    pass
    text, label  = load_data('data/back_to_en.csv')
    data = {'text':text} 
    params = {'mode':model_mode, 'multitask':multitask, 'lang':lang}
    model = SeqModel(interm_layer_size=interm_layer_size, max_length=max_length, **params)

    predict(arch, model, data, batch_size, output, images_path, weights_path, multitask=multitask)
    save_encodings(arch, model, data, batch_size, output, images_path, weights_path)
    print(f"{bcolors.OKCYAN}{bcolors.BOLD}Predictions Saved{bcolors.ENDC}")
  exit(0)