import pandas as pd
from data_utils import transpose_meta, get_ter, get_weight, get_splitted_meta
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
import random as random
import torch
import math
from collections import Counter
from constants import CHOIR_THR, PERC_THR, MAX_LEN

def filter_by_confidence(freq_conf_tensor, threshold=0.8):
  freq_conf_tensor = torch.clone(freq_conf_tensor)
  if freq_conf_tensor.ndim == 3:
    assert freq_conf_tensor.shape[1] == 2
    flattened_tensor = freq_conf_tensor.permute(0,2,1).flatten(0,1)
    flattened_tensor[flattened_tensor[:,1]<threshold, 0] = 0
    freq_conf_tensor = flattened_tensor.reshape(freq_conf_tensor.shape[0], freq_conf_tensor.shape[2], -1).permute(0,2,1)
  elif freq_conf_tensor.ndim==2:
    assert freq_conf_tensor.shape[1] == 2
    freq_conf_tensor[freq_conf_tensor[:,1]<threshold, 0] = 0 
  else:
    raise NotImplementedError
  return freq_conf_tensor


class PitchDataset:
  def __init__(self, meta, csv_dir, test_ter, min_length=15, slice_len=30, split = 'train', frame_rate=20, threshold=0.8, max_size=-1, use_pitch_aug=True):
    '''
    Args:
      meta: pd.DataFrame
      csv_dir: str, path to csv files
      test_ter
      min_length: float, minimum length of audio in seconds
      slice_len: float, length of audio slice in seconds
      split: str, 'train', 'valid', or 'test'
      frame_rate: int, frame rate of the model
      treshold: float, threshold for filtering out low confidence predictions during tonic calculation
    
    '''
    self.entire_meta = meta
    self.meta = get_splitted_meta(meta, test_ter, split)
    if max_size > 0:
      self.meta = self.meta[:max_size]
    self.csv_dir = Path(csv_dir)
    self.t_meta = transpose_meta(self.meta) 
    self.split = split
    self.frame_rate = frame_rate # Hz
    assert 100 % self.frame_rate == 0
    self.comp_ratio = 100 // self.frame_rate 
    self.min_length = min_length
    self.slice_len = slice_len
    self.use_pitch_aug = use_pitch_aug
    if not self.use_pitch_aug:
      print('Pitch augmentation is disabled.')

    self.filtered_list = self.meta[(self.meta['choir_activation']<CHOIR_THR)
                                    & (self.meta['percussion_activation']<PERC_THR)
                                    & (self.meta['length'] > self.min_length) 
                                    & (self.meta['length'] < MAX_LEN)
                                    ]['id'].tolist()
    self.meta = self.meta[self.meta['id'].isin(self.filtered_list)]

    self.splitted_csv_fn_list = self.match_id_and_csv(self.filtered_list)
    self.len_slice_frame = self.slice_len * self.frame_rate
    self.threshold = threshold

    self.length_filtered_list = self.load_all_csv() 

    self.t_meta = transpose_meta(self.meta)
    self.str2idx = {string:idx for idx, string in enumerate(get_ter())}
  
  def match_id_and_csv(self, id_list):
    csv_fn_list = [self.csv_dir/(id+'.f0.csv') for id in id_list]
    return csv_fn_list

  
  def load_all_csv(self):
    compressed_contours = [] 
    for csv in tqdm(self.splitted_csv_fn_list, leave=False):
      df = pd.read_csv(csv)
      frequency = df['frequency'].values
      confidence = df['confidence'].values
      midi = [frequency_to_midi(freq) for freq in frequency]
      tonic_counter = Counter(np.round(midi)[confidence >= self.threshold]).most_common(1)
      tonic = tonic_counter[0][0]
      norm_midi = [mi-float(tonic) for mi in midi] 
  
      freq_conf = np.stack([norm_midi, confidence], axis=0) 
      freq_conf = torch.tensor(freq_conf[:, ::self.comp_ratio], dtype=torch.float32)
      compressed_contours.append(freq_conf)
    return compressed_contours 

  def filter_by_length(self):
    length_filtered_list = []
    length_filtered_id_list = []
    for idx, csv in enumerate(self.fifth_csv_list):
      if len(csv[0]) >= (self.min_length * self.frame_rate):
        length_filtered_list.append(csv) 
        length_filtered_id_list.append(self.filtered_list[idx])
    return length_filtered_list, length_filtered_id_list
          
  def __len__(self):
    return len(self.length_filtered_list)
  
  def _random_slice_tune(self, selected_song: torch.Tensor):
    assert selected_song.ndim==2
    len_song = selected_song.shape[1]
    
    if self.split != 'train': # if valid or test, do center slice
      mid = selected_song.shape[1]//2 
      start_frame = mid- self.len_slice_frame//2 
    else: # if train, do random slice
      if len_song > (self.slice_len + 2) * self.frame_rate: # Song is long enough. 2 Sec is margin
        start_frame = random.randint(2 * self.frame_rate , len_song - 1 - self.len_slice_frame )
      elif len_song - self.len_slice_frame <= 0:
        start_frame = 0 
      else:
        start_frame = random.randint(0, len_song - 1 - self.len_slice_frame  )
    sliced_contour = selected_song[:, start_frame : start_frame + self.len_slice_frame]

    if sliced_contour.shape[1] != self.len_slice_frame: # if sliced contour is shorter than slice_len
      dummy = torch.zeros(2, self.len_slice_frame)
      middle_point = self.len_slice_frame // 2
      half_len = sliced_contour.shape[1] // 2
      dummy[:, middle_point-half_len:middle_point-half_len+sliced_contour.shape[1]] = sliced_contour
      sliced_contour = dummy

    return sliced_contour #, start_frame

  def get_label_weight(self):
    label_list = self.meta['ter'].tolist()
    return get_weight(label_list)
  
  def __getitem__(self, idx):
    selected_song = self.length_filtered_list[idx] 

    sliced_contour = self._random_slice_tune(selected_song)


    if self.use_pitch_aug and self.split=='train':
      sliced_contour[0,:] += random.random() * 12 - 6

    mp3_ter = self.meta.iloc[idx]['ter']
    return (sliced_contour, self.str2idx[mp3_ter])


def frequency_to_midi(frequency):
    # Convert frequency to MIDI note
    return 69 + 12 * math.log2(frequency / 440)




class PitchTriplet(PitchDataset):
  def __init__(self, meta, csv_dir, test_ter, min_length=15, slice_len=30, split = 'train', frame_rate=20, threshold=0.8, max_size=-1, use_pitch_aug=True):
    super().__init__(meta, csv_dir, test_ter, min_length, slice_len, split, frame_rate, threshold, max_size, use_pitch_aug=use_pitch_aug)


  def _sample_negative_tune_idx(self, anchor_idx, num_samples=5):
    assert len(self) >= num_samples + 1
    included_idx = [anchor_idx]
    cand_list = list(range(len(self)))
    cand_list.pop(anchor_idx)
    return random.sample(cand_list, num_samples)


  def __getitem__(self, idx):
    selected_song = self.length_filtered_list[idx] 
    sliced_contour = self._random_slice_tune(selected_song)
    positive_contour = self._random_slice_tune(selected_song)
    negative_ids = self._sample_negative_tune_idx(idx)
    negative_contours = [self._random_slice_tune(self.length_filtered_list[id]) for id in negative_ids]
    if self.use_pitch_aug and self.split=='train':
      sliced_contour[0,:] += random.random() * 12 - 6
      positive_contour[0,:] += random.random() * 12 - 6
      for neg_contour in negative_contours:
        neg_contour[0,:] += random.random() * 12 - 6
    return sliced_contour, positive_contour, torch.stack(negative_contours)




class ToriDataset(PitchDataset):
  def __init__(self, meta, csv_dir, frame_rate=20, min_length=20, slice_len=30, threshold=0.8, max_size=-1, load_all_csv=True, return_entire=False):
    self.meta = meta[~meta['tori'].isnull()]
    self.t_meta = transpose_meta(self.meta)
    self.id_with_tori = self.meta['id'].values.tolist()
    self.csv_dir = Path(csv_dir)
    self.splitted_csv_fn_list = [self.csv_dir / (file_id + '.f0.csv') for file_id in self.id_with_tori]
    self.frame_rate = frame_rate
    self.comp_ratio = 100 // self.frame_rate 
    self.min_length = min_length
    self.slice_len = slice_len
    self.threshold = threshold
    self.len_slice_frame = self.slice_len * self.frame_rate

    if load_all_csv:
      self.contours = self.load_all_csv()

    self.class_names = sorted(self.meta['tori'].unique().tolist())
    self.class2idx = {v:i for i, v in enumerate(self.class_names)}
    self.return_entire = return_entire
  

  def __len__(self):
    return len(self.contours)
  
  def __getitem__(self, idx):
    pitch = self.contours[idx]
    label = self.t_meta[self.id_with_tori[idx]]['tori']
    if self.return_entire:
      return pitch, self.class2idx[label]
    else:
      sliced_pitch = self._random_slice_tune(pitch)
      if self.split=='train':
        sliced_pitch[0,:] += random.random() * 12 - 6

      return sliced_pitch, self.class2idx[label]

def pad_collate(raw_batch):
  countours = [item[0] for item in raw_batch]
  labels = [item[1] for item in raw_batch]
  max_len = max([item.shape[1] for item in countours])
  padded_contours = torch.zeros(len(countours), 2, max_len)
  for i, contour in enumerate(countours):
    left = (max_len - contour.shape[1]) // 2
    padded_contours[i, :, left:left+contour.shape[1]] = contour
  return padded_contours, torch.tensor(labels)

class Monophony_filtered_Dataset(PitchDataset):
  def __init__(self, meta, csv_dir, frame_rate=20, min_length=20, slice_len=30, threshold=0.8, max_size=-1, load_all_csv=True, return_entire=False):
    self.meta = meta[(meta['choir_activation']<CHOIR_THR)
                                    & (meta['percussion_activation']<PERC_THR)
                                    & (meta['length'] > min_length)
                                    & (meta['length'] < MAX_LEN)
                                    ]
    self.t_meta = transpose_meta(self.meta)
    self.id = self.meta['id'].values.tolist()
    self.tori_meta = self.meta[~self.meta['tori'].isnull()]
    self.tori_meta_id = self.tori_meta['id'].values.tolist()
    self.csv_dir = Path(csv_dir)
    self.splitted_csv_fn_list = [self.csv_dir / (file_id + '.f0.csv') for file_id in self.id]
    self.frame_rate = frame_rate
    self.comp_ratio = 100 // self.frame_rate 
    self.min_length = min_length
    self.slice_len = slice_len
    self.threshold = threshold
    self.len_slice_frame = self.slice_len * self.frame_rate

    if load_all_csv:
      self.contours = self.load_all_csv()

    self.class_names = sorted(self.meta['ter'].unique().tolist())
    self.class2idx = {v:i for i, v in enumerate(self.class_names)}
    self.return_entire = return_entire

  def __len__(self):
    return len(self.contours)
  
  def __getitem__(self, idx):
    pitch = self.contours[idx]
    label = self.t_meta[self.id[idx]]['ter']
    if self.return_entire:
      return pitch, self.class2idx[label]
    else:
      sliced_pitch = self._random_slice_tune(pitch)
      if self.split=='train':
        sliced_pitch[0,:] += random.random() * 12 - 6

      return sliced_pitch, self.class2idx[label]
