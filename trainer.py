import torch
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb 
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
from tqdm.auto import tqdm
import torch.nn as nn
import copy
from torch.utils.data import DataLoader
import numpy as np
from model_zoo import CnnClassifier, CnnEncoder
from torch.nn.utils.rnn import pad_sequence


class Trainer:
  def __init__(self, **kwargs):
    for key, value in kwargs.items():
      setattr(self, key, value)
    self.training_loss = []
    self.valid_acc = []
    self.valid_loss = []
    self.valid_pred = []
    self.valid_label = []
    self.save_dir = Path(self.save_dir)
    self.best_model_states = None
    self.best_acc = 0
    self.iteration = 0
    self.best_downstream_acc = 0
  
    if hasattr(self, 'loss_fn') and isinstance(self.loss_fn, nn.CrossEntropyLoss):
      self.loss_fn = self.loss_fn.to(self.device)

  def train_by_num_iteration(self, num_iterations, log_name=''):
    self.model.to(self.device)
    generator = iter(self.train_loader)
    for i in tqdm(range(num_iterations)):
      try:
        # Samples the batch
        batch = next(generator)
      except StopIteration:
        # restart the generator if the previous generator is exhausted.
        generator = iter(self.train_loader)
        batch = next(generator)
      self.model.train()
      loss_value = self.train_by_single_batch(batch)
      if self.save_log:
        wandb.log({f"{log_name}training.loss": loss_value}, step=self.iteration)
      self.training_loss.append(loss_value) 

      if (i+1) % self.num_iter_per_valid == 0:
        # Do validation
        self.model.eval()
        valid_acc, valid_loss, valid_pred, valid_label= self.test_model()
        if valid_acc > self.best_acc:
          self.best_acc = valid_acc
          self.best_model_states = copy.deepcopy(self.model.state_dict())
        self.valid_acc.append(valid_acc)
        self.valid_loss.append(valid_loss)
        self.valid_pred.append(valid_pred)
        self.valid_label.append(valid_label)
        if self.save_log:
          wandb.log({f'{log_name}valid.acc': valid_acc, f'{log_name}valid.loss': valid_loss}, step=self.iteration)
        # save state to save_dir
        model_state = self.model.state_dict()
        optimizer_state = self.optimizer.state_dict()

        torch.save({'model': model_state, 'optimizer': optimizer_state}, self.save_dir / 'model_state.pt')
        if self.downstream_dataset is not None:
          downstream_result = self.do_downstream_task()
          valid_acc = downstream_result['one_split_acc']
          if valid_acc > self.best_downstream_acc:
            self.best_downstream_acc = valid_acc
            torch.save({'model': model_state, 'optimizer': optimizer_state}, self.save_dir / 'best_downstream_model_state.pt')
      self.iteration += 1
    if self.save_log and self.downstream_dataset is not None:
      wandb.run.summary['best_downstream_valid_acc'] = self.best_downstream_acc

  def train_by_num_epoch(self, num_epochs):
    for _ in tqdm(range(num_epochs)):
      self.model.train()
      for batch in tqdm(self.train_loader):
        loss_value = self.train_by_single_batch(batch)
        if self.save_log:
          wandb.log({"training.loss": loss_value})
        self.training_loss.append(loss_value) 
      self.model.eval()
      valid_acc, valid_loss, test_pred, test_label= self.test_model()
      if self.save_log:
        wandb.log({'valid.acc': valid_acc})
      self.make_confusion_matrix()
      self.valid_acc.append(valid_acc)
      self.valid_loss.append(valid_loss)
      self.valid_pred.append(test_pred)
      self.valid_label.append(test_label)

  def train_by_single_batch(self, batch):
    loss, _, _ = self.get_loss_pred_from_single_batch(batch)
    loss.backward()
    self.optimizer.step()
    self.optimizer.zero_grad()
    # if hasattr(self, 'use_zero_mean') and self.use_zero_mean:
    #   self.apply_zero_mean_conv()
    return loss.item()
  
  def get_loss_pred_from_single_batch(self, batch):
    audio, label = batch
    pred = self.model(audio.to(self.device))
    loss = self.loss_fn(pred, label.to(self.device))
    return loss, pred, label
    
  def make_confusion_matrix(self):
    self.model.eval()
    self.model.to(self.device)
    y_pred = []
    y_true = []
    ter_list = ['cb', 'cn', 'gb', 'gg', 'gn', 'gw', 'jb', 'jj', 'jn']
    with torch.no_grad():
      for batch in self.valid_loader:
        audio, label = batch
        pred = self.model(audio.to(self.device))
        y_pred+= (torch.argmax(pred.to('cpu'), dim=-1)).squeeze().tolist()
        y_true += (label.to('cpu')).squeeze().tolist()
      cm = confusion_matrix(y_true, y_pred, normalize = 'true')
      if cm.shape[0] != len(ter_list):
        print(cm)
        print('y_pred', y_pred)
        print('y_true', y_true)
        print('set', set(y_pred), set(y_true))
        return
      plt.figure(figsize=(15,10))
      df_cm = pd.DataFrame(cm, index=[ter for ter in ter_list], columns=[ter for ter in ter_list])
      
      sn.heatmap(df_cm, annot=True, fmt='.2f')
      plt.savefig(self.save_dir / f'{self.test_ter}.png')
      if self.save_log:
          wandb.log({'confusion_matrix': wandb.Image(str(self.save_dir / f'{self.test_ter}.png'))})
      plt.close()
      
  def test_model(self, loader=None):
    if loader is None:
      loader = self.valid_loader
    
    right = 0
    self.model.eval()
    self.model.to(self.device)
    accumulated_loss = 0
    all_pred = []
    all_label = []
      
    with torch.inference_mode():
      for batch in loader:
        loss, pred, label = self.get_loss_pred_from_single_batch(batch)
        opin = torch.argmax(pred, dim=-1)
        right += torch.sum(opin==label.to(self.device)) 
        accumulated_loss += loss.item() * len(label)
        all_pred += opin.tolist()
        all_label += label.to(self.device).tolist()
    valid_acc = int(right)/len(loader.dataset)
    accumulated_loss /= len(loader.dataset)
    return valid_acc, accumulated_loss, all_pred, all_label

  def do_downstream_task(self):
    dataset = self.downstream_dataset
    split_num = [int(len(dataset)*0.6), int(len(dataset)*0.2), int(len(dataset)*0.2)]
    split_num[0] += len(dataset) - sum(split_num)
    
    record = {'train.acc': [], 'train.loss': [], 'valid.acc': [], 'valid.loss': [], 'test.acc': [], 'test.loss': []}
    

    for i in range(5):
      train_set, valid_set, test_set = torch.utils.data.random_split(dataset, split_num, generator=torch.Generator().manual_seed(i))
      train_set.dataset.split = 'train'
      valid_set.dataset.split = 'valid'
      test_set.dataset.split = 'test'
      train_loader = DataLoader(train_set, batch_size=32, shuffle=True, collate_fn=pad_collate)
      valid_loader = DataLoader(valid_set, batch_size=32, shuffle=False, collate_fn=pad_collate)
      test_loader = DataLoader(test_set, batch_size=32, shuffle=False, collate_fn=pad_collate)

      trainer = DownStreamTrainer(model=self.model, 
                                  train_loader=train_loader, 
                                  valid_loader=valid_loader, 
                                  test_loader=test_loader, 
                                  save_dir=self.save_dir, 
                                  save_log=False,
                                  loss_fn=nn.CrossEntropyLoss(),
                                  device=self.device,
                                  num_iter_per_valid=100,
                                  downstream_dataset=None)
      trainer.iteration = self.iteration
      trainer.train_by_num_iteration(1000, log_name='downstream.')
      trainer.load_best_state()
      test_acc, test_loss, test_pred, test_label= trainer.test_model(test_loader)
      record['test.acc'].append(test_acc)
      record['test.loss'].append(test_loss)
      record['train.loss'].append(np.mean(trainer.training_loss[-100:]))
      record['valid.acc'].append(max(trainer.valid_acc))
      record['valid.loss'].append(min(trainer.valid_loss))

    # take average of each metric
    stat_record = {'mean':{}, 'std':{}}
    for key in record:
      stat_record['mean'][f"downstream.{key}.mean"] = np.mean(record[key])
      stat_record['std'][f"downstream.{key}.std"] = np.std(record[key])
      stat_record['one_split_acc'] = record['valid.acc'][0]

    if self.save_log:
      wandb.log(stat_record['mean'], step=self.iteration)
      wandb.log(stat_record['std'], step=self.iteration)
    return stat_record

  def load_best_state(self):
    if self.best_model_states is not None:
      self.model.load_state_dict(self.best_model_states)
      self.model.eval()
      self.model.to(self.device)
    else:
      print('No best model states to load')


class TripletTrainer(Trainer):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)

  def get_loss_pred_from_single_batch(self, batch):
    anchor, pos, neg = batch
    n = anchor.shape[0]
    stacked = torch.cat([anchor, pos, torch.flatten(neg, 0, 1)], dim=0)
    embs = self.model(stacked.to(self.device))

    anchor_emb = embs[:n]
    pos_emb = embs[n:n*2]
    neg_emb = embs[n*2:].reshape(n, -1, embs.shape[-1])
    
    loss = self.loss_fn(anchor_emb, pos_emb, neg_emb)
    return loss, None, None
  
  def test_model(self, loader=None):
    if loader is None:
      loader = self.valid_loader

    current_loss = 0
    with torch.inference_mode():
      for batch in loader:
        loss, _, _ = self.get_loss_pred_from_single_batch(batch)
        current_loss += loss.item() * len(batch[0])
      current_loss /= len(loader.dataset)
    return 0, current_loss, None, None
  

class DownStreamTrainer(Trainer):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    if isinstance(self.model, CnnClassifier):
      self.backbone = copy.deepcopy(self.model.encoder)
    elif isinstance(self.model, CnnEncoder):
      self.backbone = copy.deepcopy(self.model)
    else:
      raise NotImplementedError

    for param in self.backbone.parameters():
      param.requires_grad = False
    self.embed_size = self.backbone.embed_size


    self.training_loss = []
    self.valid_acc = []
    self.valid_loss = []
    self.valid_pred = []
    self.valid_label = []
    
    self.make_mlp_model(4)
  
  def make_mlp_model(self, num_classes):
    self.mlp = nn.Sequential(
        nn.Linear(self.embed_size, self.embed_size // 2),
        nn.ReLU(),
        nn.Linear(self.embed_size // 2, num_classes)
    )
    self.model = nn.Sequential(self.backbone, self.mlp)
    self.optimizer = torch.optim.Adam(self.mlp.parameters(), lr=0.001)
    self.model.to(self.device)


def pad_collate(batch):
  '''
  batch: list of (x, y) tuple
  x: torch.tensor of shape (feature_dim, sequence_length)
  y: int
  '''
  x, y = zip(*batch)
  x = pad_sequence([t.T for t in x], batch_first=True).permute(0, 2, 1)
  y = torch.tensor(y, dtype=torch.long)
  return x, y