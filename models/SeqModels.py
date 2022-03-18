import torch, os
from utils.params import params
import numpy as np, pandas as pd
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from utils.params import params, bcolors

class Data(Dataset):

  def __init__(self, data, fashion = 'multitask'):

    self.text = data['text']
    self.label = data['label']

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):

    if torch.is_tensor(idx):
      idx = idx.tolist()

    ret = {'text': self.text[idx]}
    
    if len(self) < 4:
      return ret
    
    if self.fashion == 'singletask':
      ret['labels'] = self.data['sexism'].iloc[idx]
    else:
      ret['labels'] = self.data[params.columns[-6:]].astype(int).iloc[idx]

    return ret

def HugginFaceLoad(language, weigths_source):

  prefix = 'data' if weigths_source == 'offline' else ''
  model = AutoModel.from_pretrained(os.path.join(prefix , params.langaugeModel[language]))
  tokenizer = AutoTokenizer.from_pretrained(os.path.join(prefix , params.langaugeModel[language]), do_lower_case=True, TOKENIZERS_PARALLELISM=True)

  return model, tokenizer
  
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
    self.model = kwargs['model']
    self.best_acc = None
    self.max_length = max_length
    self.interm_neurons = interm_size
    self.transformer, self.tokenizer = HugginFaceLoad(self.model)
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

