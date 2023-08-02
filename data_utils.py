import torch
import torchaudio
from pathlib import Path
import pickle
from collections import Counter
from tqdm import tqdm
import numpy as np
import random
import soundfile as sf
import pandas as pd
import re
import yaml
from omegaconf import OmegaConf


def get_ter():
    return ['cb', 'cn', 'gb', 'gg', 'gn', 'gw', 'jb', 'jj', 'jn']

def bring_meta(path="/home/danbi/userdata/DANBI/Korean_folk/meta_list.csv"):
    df = pd.read_csv(path)
    return df

def using_meta(path="/home/danbi/userdata/DANBI/Korean_folk/meta_list.csv"):
    wav_path = bring_wav_file()
    meta = bring_meta(path)
    title = [str(wav).split('/')[-1][:-4] for wav in wav_path]
    existing = meta['id'].isin(title)
    return meta[existing]
    # is_id = 0
    # for id_ in title:
    #     is_id = (meta['id']==id_) | is_id
    # return meta[is_id]

def get_monophony_id_list(meta):
  singer_list = list(meta['singer'])
  monophony_singer_list = filtering_monophony_from_singler_list(singer_list)
  set_singer = set(monophony_singer_list)
  monophony = meta[meta['singer'].isin(set_singer)]
  monophony_id_list = list(monophony['id'])

  return monophony_id_list


def filter_monophony(meta):
    singer_list = list(meta['singer'])
    monophony_singer_list = filtering_monophony_from_singler_list(singer_list)
    set_singer = set(monophony_singer_list)
    monophony_meta = meta[meta['singer'].isin(set_singer)]
    # monophony_id_list = list(monophony_meta['id'])
    return monophony_meta

def filtering_monophony_from_singler_list(singer_list):
  ban_word = ['외', '앞', '뒤', '가:', '나:', '앞 :', '뒤 :', '?', '장구', '장고', '상쇠', '징', '기록없음', '外', '노인회관','가;', '나;', '다;', '경북', '경기', '강원', '전북', '경로당','노인', '여럿', '전부', '남,', '짝패', '가 :', '꽹', '북:']
  filtered_singer = []
  
  num_ban = 0
  for i, singer in enumerate(singer_list):
    banned = False
    for ban in ban_word:
      if ban in str(singer):
        banned = True
        break
    if not banned:
      filtered_singer.append(singer)
  # for singer in singer_list:
  #   for ban in ban_word:
  #     if ban in str(singer):
  #       break
  # else:
  #   filtered_singer.append(singer)
    
  p = re.compile('[(].*[)].*[(]')
  refiltered_singer = []

  for singer in filtered_singer:
    if not isinstance(singer, str): # singer is nan
      continue 
    m = p.search(singer)
    if m is None:
      refiltered_singer.append(singer)
  return refiltered_singer

def transpose_meta(meta):
  idx_and_id = np.stack([meta.index.values, meta['id'].values], axis=1)
  id_meta =meta.T.rename(columns={old:new for old, new in idx_and_id})
  return id_meta
    

def bring_wav_file(path = '/home/danbi/userdata/DANBI/Korean_folk/convert_wav'):
    dir_path = Path(path)
    wav_path = list(dir_path.rglob("*.wav"))
    return wav_path

#이건 필요없는거(예전꺼)
def load_ter_dic():
    with open('big_ter_dic.pkl', 'rb') as f:
        ter_dic = pickle.load(f)
    return ter_dic

def detail_ter(dir_path):
    mp3_list = list(dir_path.rglob("*.mp3"))
    detail_ter_list = []
    for i in range(len(mp3_list)):
        detail_ter_list.append(str(mp3_list[i]).split('/')[-1][:-9])
    return list(set(detail_ter_list))

def test_ter(select_num=16, seed = 0):
    random.seed(seed)
    det_ter = list(set(using_meta()['det_ter']))
    test_idx_list = []
    test_ter = []
    num = 0
    while True:
        idx = random.randint(1, len(det_ter)-1)
        if idx not in test_idx_list:
            test_idx_list.append(idx)
            test_ter.append(det_ter[idx])
            num+=1
        if num == select_num:
            break
    return test_ter


def save_len(wav):
  ob = sf.SoundFile(wav) 
  duration = ob.frames / ob.samplerate
  return float(duration)

def how_many(wav_list, test_ter):
  num_test = []
  for i in range(len(wav_list)):
    if str(wav_list[i]).split('/')[-1][:-9] in test_ter:
      num_test.append(wav_list[i])
  return len(num_test)

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


class KoreanFolk:
  def __init__(self, meta, dir_path, test_ter, min_length, max_length, split='train'):
    self.splitted_meta = get_splitted_meta(meta, test_ter, split)
    self.t_meta = transpose_meta(self.meta)
    self.dir_path = Path(dir_path)
    self.wav_list = list(self.dir_path.rglob('*.wav'))
    self.meta_list = list(self.splitted_meta['id'])
    self.min_length, self.max_length = min_length, max_length
    self.wav_list = self.len_filter()
    self.wav_pths = []
    self.song_list = []
    for wav in self.wav_list:
        string = str(wav).split('/')[-1][:-4]
        if string in self.meta_list and wav not in self.wav_pths:
            self.wav_pths.append(wav)
            self.song_list.append(string)
    self.str2idx = {string:idx for idx, string in enumerate(get_ter())}
    self.label_weight = self.get_label_weight()
  
  def len_filter(self):
    filtered_list = []
    for wav in tqdm(self.wav_list):
      if self.min_length <= save_len(wav) <= self.max_length:
        filtered_list.append(wav)
    return filtered_list
  
  def __len__(self):
    return len(self.wav_pths)
  
  def get_label_weight(self):
    label_list = []
    for i in range(len(self.song_list)):
      label = self.t_meta[self.song_list[i]]['ter']
      label_list.append(label)
    return get_weight(label_list)
  
  def __getitem__(self, idx):
    pth = self.wav_pths[idx]
    y, sr = torchaudio.load(pth)
    _len = int(save_len(pth))
    start_sec = random.randint(2, _len-15)
    mp3_ter = self.t_meta[self.song_list[idx]]['ter']
    audio = y[:,(start_sec*sr):((start_sec+15)*sr)].squeeze(dim=0)
#         print(audio.shape)
    return audio, self.str2idx[mp3_ter]

def random_test_ter(meta=None, random_seed = 0):
    ter_dic = {}
    random.seed(random_seed)
    test_ter = []
    if meta is None:
        meta = using_meta()
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


# class OldKoreanFolk:
#     def __init__(self, dir_path, ter_dic, min_length, max_length, test_ter, split='train'):
#         self.dir_path = Path(dir_path)
#         self.wav_list = list(self.dir_path.rglob('*.wav'))
#         self.min_length, self.max_length = min_length, max_length
#         self.wav_pths = self.len_filter()
#         self.ter_list = list(set(ter_dic.values()))
#         self.ter_list.sort()
#         self.str2idx = {string:idx for idx, string in enumerate(self.ter_list)}
#         self.ter_dic = ter_dic
#         self.test_ter = test_ter
#         train_list = []
#         test_list = []
#         for i in range(len(self.wav_pths)):
#             if str(self.wav_pths[i]).split('/')[-1][:-9] in self.test_ter:
#                 test_list.append(self.wav_pths[i])
#             else:
#                 train_list.append(self.wav_pths[i])
        
#         if split == 'train':
#             self.wav_pths = train_list
#         if split == 'test':
#             self.wav_pths = test_list
        
#     def len_filter(self):
#         filtered_list = []
#         for wav in tqdm(self.wav_list):
#             if self.min_length <= save_len(wav) <= self.max_length:
#                 filtered_list.append(wav)
#         return filtered_list
    
#     def get_label(self):
#         label_list = []
#         for i in range(len(self.wav_pths)):
#             mp3_ter = self.ter_dic[(str(self.wav_pths[i]).split('/'))[-1][:-4]+'.mp3']
#             label_list.append(self.str2idx[mp3_ter])
#         return label_list
    
#     def __len__(self):
#         return len(self.wav_pths)
     
#     def __getitem__(self, idx):
#         pth = self.wav_pths[idx]
#         y, sr = torchaudio.load(pth)
#         _len = int(save_len(pth))
#         start_sec = random.randint(2, _len-15)
#         mp3_ter = self.ter_dic[(str(pth).split('/'))[-1][:-4]+'.mp3']
#         audio = y[:,(start_sec*sr):((start_sec+15)*sr)].squeeze(dim=0)
# #         print(audio.shape)
#         return audio, self.str2idx[mp3_ter]
