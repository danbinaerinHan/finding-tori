import torch
from pathlib import Path
from omegaconf import OmegaConf
import pandas as pd
from data_utils import read_yaml, wandb_style_config_to_omega_config
from model_zoo import CnnClassifier, CnnEncoder
from pitch_utils import PitchTriplet, PitchDataset, ToriDataset, pad_collate
import random
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import argparse
from collections import Counter
from handfeatures import frequency_to_midi, get_midi_contour_from_csv, HistogramMaker

CONTOUR_DIR = Path('contour_csv/')


def get_cosine_similarity_matrix(emb):
  '''
  emb: torch.Tensor (N, D) # 5752, 512
  '''
  emb_norm = emb / emb.norm(dim=1, keepdim=True)
  sim =  emb_norm @ emb_norm.T
  # make diagonal 0
  sim = sim - torch.eye(len(emb)).to(sim.device)
  return sim

def get_topk_similar_indices(similarity_mat, k=10):
  '''
  similarity_mat: torch.Tensor (N, N)
  k: int
  '''
  topk_similar_indices = torch.topk(similarity_mat, k=k, dim=1)[1]
  return topk_similar_indices

def get_ndcg(scores):
  '''
  scores: torch.Tensor (N, )
  '''
  dcg = scores / torch.log2(torch.arange(len(scores), dtype=torch.float) + 2)
  idcg = torch.sort(scores, descending=True)[0] / torch.log2(torch.arange(len(scores), dtype=torch.float) + 2)
  return (dcg.sum() / idcg.sum()).item()

def check_k_fold_random_forest(embeddings, labels, k=5, num_train_ratio=0.75):
  if isinstance(labels, torch.Tensor):
    labels = labels.tolist()
  if isinstance(embeddings, torch.Tensor):
    embeddings = embeddings.numpy()
  accs = []
  num_train_samples = int(len(embeddings) * num_train_ratio)
  wrong_samples = []
  test_samples_ids = []

  for i in range(k):
    random.seed(i)
    rand_train_idx = random.sample(range(len(embeddings)), num_train_samples)
    rand_test_idx = [idx for idx in range(len(embeddings)) if idx not in rand_train_idx]
    train_hists = [embeddings[idx] for idx in rand_train_idx]
    train_labels = [labels[idx] for idx in rand_train_idx]

    test_hists = [embeddings[idx] for idx in range(len(embeddings)) if idx in rand_test_idx]
    test_labels = [labels[idx] for idx in range(len(embeddings)) if idx in rand_test_idx]

    rf_classifier = RandomForestClassifier()
    rf_classifier.fit(train_hists, train_labels)
    test_pred = rf_classifier.predict(test_hists)
    wrong_idx = [idx for idx in range(len(test_pred)) if test_pred[idx] != test_labels[idx]]
    entire_wrong_idx = [rand_test_idx[idx] for idx in wrong_idx]
    wrong_samples += entire_wrong_idx
    score = rf_classifier.score(test_hists, test_labels)
    test_samples_ids += rand_test_idx

    accs.append(score)
  wrong_counter = Counter(wrong_samples)
  test_counter = Counter(test_samples_ids)
  # divide wrong counter by test counter
  for k in wrong_counter.keys():
    wrong_counter[k] /= test_counter[k]
  return np.mean(accs), np.std(accs), wrong_counter


def get_args():
  # get ckpt path as argument
  parser = argparse.ArgumentParser()
  parser.add_argument('--ckpt_path', type=str)
  parser.add_argument('--model', type=str, default='self-supervised', choices=['self-supervised', 'region-supervised'])
  parser.add_argument('--wandb_dir', type=str, default='wandb')
  parser.add_argument('--use_histogram', action='store_true')
  parser.add_argument('--resolution', type=float, default=0.2)


  args = parser.parse_args()

  return args


def get_dir_from_wandb_by_code(wandb_dir, code):
  for dir in wandb_dir.iterdir():
    if dir.name.endswith(code):
      return dir
  return None

def get_best_ckpt_path(wandb_dir, code):
  dir = get_dir_from_wandb_by_code(wandb_dir, code)
  if dir is None:
    raise ValueError('No such code in wandb_dir')
  ckpt_path = dir / 'files' / 'checkpoints' / 'model_state.pt'
  if not ckpt_path.exists():
    raise ValueError('No best ckpt in wandb_dir')
  return ckpt_path

def get_embedding_from_model(ckpt_path):
  print(f'ckpt_path: {ckpt_path}')
  yaml_path = ckpt_path.parent / 'config.yaml'
  config = read_yaml(yaml_path)
  if not hasattr(config, 'meta_csv_path'):
    config = wandb_style_config_to_omega_config(config)
  print(f'config: {config}')
  meta = pd.read_csv(config.meta_csv_path)

  if config.exp == 'self_supervised':
    model = CnnEncoder(config.model_params)
  else:
    model = CnnClassifier(config.model_params)

  model.load_state_dict(torch.load(ckpt_path)['model'])
  model.eval()

  if isinstance(model, CnnClassifier):
    model = model.encoder
  DEV = 'cuda'
  model.to(DEV)

  tori_set = ToriDataset(meta, CONTOUR_DIR, frame_rate=config.train.frame_rate, min_length=config.train.min_length, slice_len=config.train.slice_len, return_entire=True)

  data_loader = torch.utils.data.DataLoader(tori_set, batch_size=1, shuffle=False, collate_fn=pad_collate)

  embeddings = []
  labels = []
  with torch.inference_mode():
    for i, (x, y) in enumerate(data_loader):
      embedding = model(x.to(DEV))
      embeddings.append(embedding)
      labels.append(y)
  embeddings = torch.cat(embeddings, dim=0).cpu()
  labels = torch.tensor(labels).cpu().squeeze()
  
  return tori_set, embeddings, labels

def get_embedding_from_histogram(resolution=0.2):
  print('get embedding from histogram')
  config = read_yaml('yamls/baseline.yaml')
  meta = pd.read_csv(config.meta_csv_path)
  tori_set = ToriDataset(meta, CONTOUR_DIR, frame_rate=config.train.frame_rate, min_length=config.train.min_length, slice_len=config.train.slice_len, return_entire=True)
  csv_fns = tori_set.splitted_csv_fn_list
  midi_contours = [get_midi_contour_from_csv(x) for x in csv_fns]
  hist_maker = HistogramMaker(resolution=resolution)
  print(f'making histogram using {len(hist_maker.bins)} bins ')
  norm_hists = [hist_maker(midi_contour) for midi_contour in midi_contours]
  norm_hists = torch.tensor(norm_hists)

  labels = torch.tensor([tori_set.class2idx[x] for x in tori_set.meta['tori']])
  return tori_set, norm_hists, labels

def main():
  args = get_args()
  if args.use_histogram:
    tori_set, embeddings, labels = get_embedding_from_histogram(args.resolution)
  else:
    if args.ckpt_path is not None:
      ckpt_path = Path(args.ckpt_path)
    elif args.model == 'self-supervised':
      ckpt_path = Path('pretrained_weights/self-supervised/model_state.pt')
    elif args.model == 'region-supervised':
      ckpt_path = Path('pretrained_weights/region-supervised/model_state.pt')
    else:
      raise ValueError('model should be either self-supervised or region-supervised, or provide ckpt_path')
    tori_set, embeddings, labels = get_embedding_from_model(ckpt_path)

  similarity_mat = get_cosine_similarity_matrix(embeddings)
  top_k_indices = get_topk_similar_indices(similarity_mat, k=len(embeddings))

  total_ndcg = 0
  ndcg_per_label = [0, 0, 0, 0]
  num_sample_per_label = [0, 0, 0, 0]
  for idx in range(len(top_k_indices)):
    is_same_tori = (labels[idx] == labels[top_k_indices[idx]]).float()
    ndcg = get_ndcg(is_same_tori)
    total_ndcg += ndcg
    ndcg_per_label[labels[idx]] += ndcg
    num_sample_per_label[labels[idx]] += 1
  ndcg_per_label = [ndcg_per_label[i] / num_sample_per_label[i] for i in range(4)]
  print(f"Label names are {tori_set.class_names}")
  print(f"Mean NDCG: {total_ndcg / len(top_k_indices)}, NDCG per label: {ndcg_per_label}")


  is_not_speech = (labels != tori_set.class2idx['others']).float()
  emb_exc_speech = embeddings[is_not_speech == 1]
  labels_exc_speech = labels[is_not_speech == 1]
  similarity_mat_exc_speech = get_cosine_similarity_matrix(emb_exc_speech)
  top_k_indices_exc_speech = get_topk_similar_indices(similarity_mat_exc_speech, k=len(emb_exc_speech))

  total_ndcg = 0
  ndcg_per_label = [0, 0, 0, 0]
  num_sample_per_label = [0, 0, 1, 0]
  for idx in range(len(top_k_indices_exc_speech)):
    is_same_tori = (labels_exc_speech[idx] == labels_exc_speech[top_k_indices_exc_speech[idx]]).float()
    ndcg = get_ndcg(is_same_tori)
    total_ndcg += ndcg
    ndcg_per_label[labels_exc_speech[idx]] += ndcg
    num_sample_per_label[labels_exc_speech[idx]] += 1
  ndcg_per_label = [ndcg_per_label[i] / num_sample_per_label[i] for i in range(4)]
  print(f"Label names are {tori_set.class_names}")
  print(f"Mean NDCG: {total_ndcg / len(top_k_indices_exc_speech)}, NDCG per label: {ndcg_per_label}")


  k_mean, k_std, wrong_samples = check_k_fold_random_forest(embeddings, labels, k=30)
  print(f"Random forest accuracy: {k_mean} +- {k_std}")

  k_mean, k_std, wrong_samples = check_k_fold_random_forest(emb_exc_speech, labels_exc_speech, k=30)
  print(f"Random forest accuracy (without others): {k_mean} +- {k_std}")

if __name__ == '__main__':
  main()