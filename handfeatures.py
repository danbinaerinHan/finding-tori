import pandas as pd
from pitch_utils import PitchDataset, ToriDataset
import math
import numpy as np
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
import random

DIR = "contour_csv/"

def frequency_to_midi(frequency):
  # Convert frequency to MIDI note
  return 69 + 12 * math.log2(frequency / 440)

def get_midi_contour_from_csv(csv_fn, low_pitch=40, high_pitch=72):
  df = pd.read_csv(csv_fn)
  frequency = df['frequency'].values
  confidence = df['confidence'].values
  threshold = 0.8
  frequency[confidence < threshold] = np.nan
  pitch_in_midi = [frequency_to_midi(freq) for freq in frequency]
  return [x for x in pitch_in_midi if low_pitch <= x < high_pitch]

def get_norm_pitch_histogram(midi_contour, compensate_tuning=True):
  '''
  midi_contour: list of midi pitch as a contour
  '''
  edge = [i+(-12.5) for i in range(26)]
  max_appearance = 0
  comp = 0
  final_common_midi = 0
  if compensate_tuning:
    for compensate in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
      intmidi = [round(midi+compensate) for midi in midi_contour]
      common_pitch, num_appearance = Counter(intmidi).most_common()[0]
      if max_appearance <= num_appearance:
        max_appearance = num_appearance
        comp = compensate
        final_common_midi = common_pitch
  else:
    intmidi = [round(midi) for midi in midi_contour]
    common_pitch, num_appearance = Counter(intmidi).most_common()[0]
    final_common_midi = common_pitch
  norm_midi = [midi-float(final_common_midi-comp) for midi in midi_contour]
  mono_hist = np.histogram(norm_midi, bins = edge, density=True)
  return mono_hist[0]

class HistogramMaker:
  def __init__(self, resolution=0.2, compensate_tuning=True):
    self.resolution = resolution
    self.edges, self.bins = self.make_edges(resolution)
    self.comp_tuning = compensate_tuning
  
  def make_edges(self, resolution):
    edges = [ ]
    lowest_interval = -12.5
    highest_interval = 12.5

    lowest_edge = lowest_interval - resolution / 2
    highest_edge = highest_interval + resolution / 2

    for i in range(int((highest_edge - lowest_edge) / resolution)):
      edges.append(lowest_edge + resolution * i)    

    bins = np.array(edges) + resolution / 2
    bins = bins[:-1]

    return edges, bins
  
  def get_tonic(self, midi_contour, compensate_tuning):
    max_appearance = 0
    comp = 0
    final_common_midi = 0
    if compensate_tuning:
      for compensate in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        intmidi = [round(midi+compensate) for midi in midi_contour]
        common_pitch, num_appearance = Counter(intmidi).most_common()[0]
        if max_appearance <= num_appearance:
          max_appearance = num_appearance
          comp = compensate
          final_common_midi = common_pitch
    else:
      intmidi = [round(midi) for midi in midi_contour]
      common_pitch, num_appearance = Counter(intmidi).most_common()[0]
      final_common_midi = common_pitch
    final_common_midi = final_common_midi - comp
    return final_common_midi
  

  def get_norm_pitch_histogram(self, midi_contour, compensate_tuning=True, edge=None):
    '''
    midi_contour: list of midi pitch as a contour
    '''
    if edge is None :
      edge = [i+(-12.5) for i in range(26)]
    tonic = self.get_tonic(midi_contour, compensate_tuning)
    # print(f"Most frequently appearing: {final_common_midi}")
    norm_midi = [midi-float(tonic) for midi in midi_contour]
    mono_hist = np.histogram(norm_midi, bins = edge, density=True)
    return mono_hist[0]

  def __call__(self, contour):
    return self.get_norm_pitch_histogram(contour, self.comp_tuning, self.edges)


def main():
  # read meta
  meta = pd.read_csv("metadata.csv")

  dataset = ToriDataset(meta, DIR, load_all_csv=False)
  csv_fns = dataset.splitted_csv_fn_list

  midi_contours = [get_midi_contour_from_csv(x) for x in csv_fns]
  labels = [dataset.t_meta[dataset.id_with_tori[idx]]['tori'] for idx in range(len(csv_fns))]
  norm_hists = [get_norm_pitch_histogram(midi_contour, compensate_tuning=False) for midi_contour in midi_contours]

  num_train_samples = 150

  accs = []

  for i in range(10):
    random.seed(i)

    rand_train_idx = random.sample(range(len(norm_hists)), num_train_samples)
    train_hists = [norm_hists[idx] for idx in rand_train_idx]
    train_labels = [labels[idx] for idx in rand_train_idx]

    test_hists = [norm_hists[idx] for idx in range(len(norm_hists)) if idx not in rand_train_idx]
    test_labels = [labels[idx] for idx in range(len(norm_hists)) if idx not in rand_train_idx]

    rf_classifier = RandomForestClassifier()
    rf_classifier.fit(train_hists, train_labels)
    score = rf_classifier.score(test_hists, test_labels)
    accs.append(score)

  print(accs)
  print(np.mean(accs), np.std(accs))
  return 


if __name__ == "__main__":
  main()