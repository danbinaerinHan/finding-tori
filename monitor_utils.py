from pathlib import Path
import pandas as pd
import numpy as np
import torchaudio
import IPython.display as ipd
import matplotlib.pyplot as plt
from IPython.display import display

class DataMonitor:
  def __init__(self, csv_dir, threshold, sr, audio_path):
    self.csv_dir = Path(csv_dir)
    self.csv_fns = sorted(list(self.csv_dir.rglob('*.csv')))
    self.threshold = threshold
    self.sr = sr
    self.audio_path = audio_path

  def get_frequency(self, idx_or_fn):
    if isinstance(idx_or_fn, int):
      csv_fn = self.csv_fns[idx_or_fn]
    else:
      csv_fn = Path(idx_or_fn)

    df = pd.read_csv(csv_fn)
    frequency = df['frequency'].values
    confidence = df['confidence'].values
    frequency[confidence<self.threshold] = np.nan
    return frequency
  
  def get_frequency_figure(self):
    plt.figure(figsize = (15, 7))
    plt.plot(self.frequency[:5000])
    return plt.show()
  
  def get_audio(self, idx_or_fn):
    if isinstance(idx_or_fn, int):
      csv_fn = self.csv_fns[idx_or_fn]
    else:
      csv_fn = Path(idx_or_fn)
    audio_name = csv_fn.stem[:-3]
    audio_fn = f"{self.audio_path}/{audio_name}.wav"
    y, sr = torchaudio.load(audio_fn)
    audio = y.numpy()[0]
    return display(ipd.Audio(y, rate=self.sr)), audio
  
  def generate_sine_wav(self, idx_or_fn):
    if isinstance(idx_or_fn, int):
      csv_fn = self.csv_fns[idx_or_fn]
    else:
      csv_fn = Path(idx_or_fn)
    frequency = self.get_frequency(idx_or_fn)
    audio = self.get_audio(idx_or_fn)[1]
    
    zero_replaced_frequency = np.asarray(frequency)
    zero_replaced_frequency[np.isnan(zero_replaced_frequency)] = 0
    melody_resampled = np.repeat(zero_replaced_frequency, self.sr//100)
    phi = np.zeros_like(melody_resampled)
    phi[1:] = np.cumsum(2* np.pi * melody_resampled[:-1] / self.sr, axis=0)
    sin_wav = 0.9 * np.sin(phi)
    # plt.plot(sin_wav)
    sin_wav = sin_wav[:audio.shape[0]]
    syn_wav = sin_wav+(audio*1)
    return display(ipd.Audio(syn_wav, rate=self.sr)), csv_fn.stem[:-3]
 