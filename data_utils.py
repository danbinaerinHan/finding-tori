import torch
from collections import Counter
import numpy as np
import random
import yaml
from omegaconf import OmegaConf


def get_ter():
    return ['cb', 'cn', 'gb', 'gg', 'gn', 'gw', 'jb', 'jj', 'jn']


def transpose_meta(meta):
  idx_and_id = np.stack([meta.index.values, meta['id'].values], axis=1)
  id_meta =meta.T.rename(columns={old:new for old, new in idx_and_id})
  return id_meta
    


def get_weight(label_list):
  ter_ratio = []
  count = Counter(label_list)
  for ter in (get_ter()):
    ratio = (1/len(get_ter())) / (count[ter]/len(label_list))
    ter_ratio.append(ratio)
  return torch.Tensor(ter_ratio)


def get_splitted_meta(meta, test_ter, split='train'):
  if split == 'test':    
    test_meta = meta.loc[meta['det_ter'].isin(test_ter)]
    return test_meta
  elif split == 'train':
    train_meta = meta.loc[meta['det_ter'].isin(test_ter)==False] 
    return train_meta
  else:
    raise NotImplementedError


def random_test_ter(meta, random_seed = 0):
    random.seed(random_seed)
    test_ter = []
    for ter in get_ter():
        unmeta = meta[meta['ter']==ter]
        det_ter = list(set(unmeta['det_ter']))
        test_idx_list = []
        for i in range(len(det_ter)//10+1):
            idx = random.randrange(len(det_ter))
            if idx not in test_idx_list:
                test_idx_list.append(idx)
                test_ter.append(det_ter[idx])

    return test_ter


def read_yaml(yml_path):
  with open(yml_path, 'r') as f:
    yaml_obj = yaml.load(f, Loader=yaml.FullLoader)
  config = OmegaConf.create(yaml_obj)
  return config


def wandb_style_config_to_omega_config(wandb_conf):
    # remove wandb related config
    for wandb_key in ["wandb_version", "_wandb"]:
      if wandb_key in wandb_conf:
        del wandb_conf[wandb_key] # wandb-related config should not be overrided! 

    # remove nonnecessary fields such as desc and value
    for key in wandb_conf:
        if 'desc' in wandb_conf[key]:
            del wandb_conf[key]['desc']
        if 'value' in wandb_conf[key]:
            wandb_conf[key] = wandb_conf[key]['value']

    return wandb_conf