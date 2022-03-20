from pytest import param
import torch, os, random
from utils.params import params
import numpy as np, pandas as pd
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from utils.params import params, bcolors
from sklearn.model_selection import StratifiedKFold

class Data(Dataset):

  def __init__(self, data, fashion):

    self.text = data['text']
    self.label = data['label']
    self.fashion = fashion

  def __len__(self):
    return len(self.text)

  def __getitem__(self, idx):

    if torch.is_tensor(idx):
      idx = idx.tolist()

    ret = {'text': self.text[idx]}
    
    if len(self) < 4:
      return ret
    
    if self.fashion == 'singletask':
      ret['labels'] = self.label[idx, 0]
    else:
      ret['labels'] = self.label[idx]

    return ret

def HugginFaceLoad(language, weigths_source):

  prefix = 'data' if weigths_source == 'offline' else ''
  model = AutoModel.from_pretrained(os.path.join(prefix , params.models[language]))
  tokenizer = AutoTokenizer.from_pretrained(os.path.join(prefix , params.models[language]), do_lower_case=True, TOKENIZERS_PARALLELISM=True)

  return model, tokenizer

def seed_worker(worker_id):
  worker_seed = torch.initial_seed() % 2**32
  np.random.seed(worker_seed)
  random.seed(worker_seed)

class MultiTaskLoss(torch.nn.Module):
    def __init__(self):
        super(MultiTaskLoss, self).__init__()

    def sigmoid(self, z ):
      return 1./(1 + torch.exp(-z))

    def forward(self, outputs, labels):

        outputs = self.sigmoid(outputs) 
        outputs = (-(labels*torch.log(outputs) + (1. - labels)*torch.log(1. - outputs))*torch.where(labels == -1, 0, 1)).sum(axis=-1)
        
        return outputs.mean()

class SeqModel(torch.nn.Module):

  def __init__(self, interm_size, max_length, **kwargs):

    super(SeqModel, self).__init__()
		
    self.mode = kwargs['mode']
    self.best_acc = None
    self.max_length = max_length
    self.interm_neurons = interm_size
    self.transformer, self.tokenizer = HugginFaceLoad( kwargs['lang'], self.mode)
    self.intermediate = torch.nn.Sequential(torch.nn.Linear(in_features=768, out_features=self.interm_neurons), torch.nn.LeakyReLU(),
                                            torch.nn.Linear(in_features=self.interm_neurons, out_features=self.interm_neurons>>1),
                                            torch.nn.LeakyReLU())
    
    if kwargs['multitask'] == True:
      self.classifier = torch.nn.Linear(in_features=self.interm_neurons>>1, out_features=6)
      self.loss_criterion = MultiTaskLoss()
    else: 
      self.classifier = torch.nn.Linear(in_features=self.interm_neurons>>1, out_features=2)
      self.loss_criterion = torch.nn.CrossEntropyLoss()
    
    self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    self.to(device=self.device)

  def forward(self, data, get_encoding=False):

    ids = self.tokenizer(data['text'], return_tensors='pt', truncation=True, padding=True, max_length=self.max_length).to(device=self.device)

    X = self.transformer(**ids)[0]

    X = X[:,0]
    enc = self.intermediate(X)
    output = self.classifier(enc)
    if get_encoding == True:
      return enc

    return output 

  def load(self, path):
    self.load_state_dict(torch.load(path, map_location=self.device))

  def save(self, path):
    torch.save(self.state_dict(), path)

  def makeOptimizer(self, lr=1e-5, decay=2e-5, multiplier=1, increase=0.1):

    if self.mode == 'static':
      return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=decay)

    params = []
    for l in self.transformer.encoder.layer:

      params.append({'params':l.parameters(), 'lr':lr*multiplier}) 
      multiplier += increase

    try:
      params.append({'params':self.transformer.pooler.parameters(), 'lr':lr*multiplier})
    except:
      print(f'{bcolors.WARNING}Warning: No Pooler layer found{bcolors.ENDC}')

    params.append({'params':self.intermediate.parameters(), 'lr':lr*multiplier})
    params.append({'params':self.classifier.parameters(), 'lr':lr*multiplier})

    return torch.optim.RMSprop(params, lr=lr*multiplier, weight_decay=decay)

def sigmoid( z ):
  return 1./(1 + torch.exp(-z))

def compute_acc(ground_truth, predictions, multitask):

  if multitask == False:
    return((1.0*(torch.max(predictions, 1).indices == ground_truth)).sum()/len(ground_truth)).cpu().numpy()

  predictions = torch.where(sigmoid(predictions) > 0.5, 1, 0)

  acc = []
  for i in range(ground_truth.shape[1]):
    acc.append( ((1.0*(predictions[:,i] == ground_truth[:,i])).sum()/ground_truth.shape[0]).cpu().numpy() )
  return np.array(acc)

def train_model(model_name, model, trainloader, devloader, epoches, lr, decay, output, split=1, multitask=False):
  
  eloss, eacc, edev_loss, edev_acc = [], [], [], []

  optimizer = model.makeOptimizer(lr=lr, decay=decay)
  batches = len(trainloader)

  for epoch in range(epoches):

    running_loss = 0.0
    perc = 0
    acc = 0
    
    model.train()
    last_printed = ''

    for j, data in enumerate(trainloader, 0):

      torch.cuda.empty_cache()         
      labels = data['labels'].to(model.device)     
      
      optimizer.zero_grad()
      outputs = model(data)
      loss = model.loss_criterion(outputs, labels)
   
      loss.backward()
      optimizer.step()

      # print statistics
      with torch.no_grad():
        if j == 0:
          acc = compute_acc(labels, outputs, multitask)
          running_loss = loss.item()
        else: 
          acc = (acc + compute_acc(labels, outputs, multitask))/2.0
          running_loss = (running_loss + loss.item())/2.0

      if (j+1)*100.0/batches - perc  >= 1 or j == batches-1:
        
        perc = (1+j)*100.0/batches
        last_printed = f'\rEpoch:{epoch+1:3d} of {epoches} step {j+1} of {batches}. {perc:.1f}% loss: {running_loss:.3f}'
        
        print(last_printed , end="")#+ compute_eta(((time.time()-start_time)*batches)//(j+1))

    model.eval()
    eloss.append(running_loss)
    with torch.no_grad():
      out = None
      log = None
      for k, data in enumerate(devloader, 0):
        torch.cuda.empty_cache() 
        labels = data['labels'].to(model.device) 

        dev_out = model(data)
        if k == 0:
          out = dev_out
          log = labels
        else: 
          out = torch.cat((out, dev_out), 0)
          log = torch.cat((log, labels), 0)

      dev_loss = model.loss_criterion(out, log).item()
      dev_acc = compute_acc(log, out, multitask)
      eacc.append(acc)
      edev_loss.append(dev_loss)
      edev_acc.append(dev_acc) 

    band = False

    measure = dev_acc
    if multitask == True:
      measure = dev_acc[0]

    if model.best_acc is None or model.best_acc < measure:
      model.save(os.path.join(output, f'{model_name}_split_{split}.pt'))
      model.best_acc = measure
      band = True

    # ep_finish_print = f' acc: {acc:.3f} | dev_loss: {dev_loss:.3f} dev_acc: {dev_acc.reshape(-1)[0]:.3f}'
    ep_finish_print = f' acc: {acc} | dev_loss: {dev_loss:.3f} dev_acc: {dev_acc.reshape(-1)}'

    if band == True:
      print(bcolors.OKBLUE + bcolors.BOLD + last_printed + ep_finish_print + '\t[Weights Updated]' + bcolors.ENDC)
    else: print(last_printed + ep_finish_print)

  return {'loss': eloss, 'acc': eacc, 'dev_loss': edev_loss, 'dev_acc': edev_acc}


def train_model_CV(model_name, lang, data, splits = 5, epoches = 4, batch_size = 8, max_length = 120, 
                    interm_layer_size = 64, lr = 1e-5,  decay=2e-5, output='logs', multitask=False, model_mode='offline'):

  history = []
  skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state = 23)
  
  tmplb = data['labels'][:,0]
  params = {'mode':model_mode, 'multitask':multitask, 'lang':lang}
  for i, (train_index, test_index) in enumerate(skf.split(data['text'], tmplb)):  
    
    history.append({'loss': [], 'acc':[], 'dev_loss': [], 'dev_acc': []})
    model = SeqModel(interm_layer_size, max_length, **params)

    trainloader = DataLoader(Data({'text':data['text'][train_index], 'label': data['labels'][train_index]}, 'singletask' if not multitask else 'multitask'), batch_size=batch_size, shuffle=True, num_workers=4, worker_init_fn=seed_worker)
    devloader = DataLoader(Data({'text':data['text'][test_index], 'label':data['labels'][test_index]}, 'singletask' if not multitask else 'multitask'), batch_size=batch_size, shuffle=True, num_workers=4, worker_init_fn=seed_worker)

    history.append(train_model(model_name, model, trainloader, devloader, epoches, lr, decay, output, i+1, multitask=multitask))
      
    print('Training Finished Split: {}'. format(i+1))
    del trainloader
    del model
    del devloader
    break
  return history

def predict(model_name, model, data, batch_size, output, images_path, wp,  multitask = False, split = 1):
  devloader = DataLoader(Data(data), batch_size=batch_size, shuffle=False, num_workers=4, worker_init_fn=seed_worker)
  model.eval()
  model.load(os.path.join(wp, f'{model_name}_split_{split}.pt'))
  with torch.no_grad():
    out = None
    for k, data in enumerate(devloader, 0):   
      dev_out = model(data)
      if k == 0:
          out = dev_out
      else:  out = torch.cat((out, dev_out), 0)
  
  if os.path.isdir(output) == False:
      os.system(f'mkdir {output}')

  if multitask == False:
    y_hat = np.int32(np.round(torch.argmax(torch.nn.functional.softmax(out, dim=-1), axis=-1).cpu().numpy(), decimals=0))
    dictionary = {'id': np.array([i.split('/')[-1] for i in images_path]),  'misogynous':y_hat}  
    df = pandas.DataFrame(dictionary) 
    df.to_csv(os.path.join(output, 'preds.csv'), sep='\t', index=False, header=False)
  else:
    y_hat = torch.where(sigmoid(out) > 0.5, 1, 0).cpu().numpy()
    dictionary = {'id': np.array([i.split('/')[-1] for i in images_path]),  'misogynous':y_hat[:,0], 'shaming':y_hat[:,1],	'stereotype':y_hat[:,2], 'objectification':y_hat[:,3],	'violence': y_hat[:,4]}  
    df = pandas.DataFrame(dictionary) 
    df.to_csv(os.path.join(output, 'preds.csv'), sep='\t', index=False, header=False)
