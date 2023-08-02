import torch
import torchaudio
import torch.nn as nn

def get_nll_loss(pred, target, weight, eps =1e-6, is_test=False):    
    pred_value = pred[torch.arange(len(target)), target]
    loss = -torch.log(pred_value+eps)
    if is_test:
        pass
    else: 
        new_weight = weight[target]
        loss = torch.mean(loss.mul(new_weight))
#     for i in range(32):
#         _wei = weight[int(target[i])]
#         new_weight.append(_wei)
           
    return loss

def get_test_loss(pred, target, eps = 1e-6):
    pred_value = pred[torch.arange(len(target)), target]
    loss = -torch.log(pred_value+eps)
    return loss

class MarginHingeLoss(nn.Module):
  def __init__(self, margin=0.4) -> None:
    super().__init__()
    self.margin = margin
    self.cos = nn.CosineSimilarity(dim=2, eps=1e-6)

  def forward(self, anchor, pos, neg):
    assert anchor.ndim == 2 and neg.ndim == 3
    anchor = anchor.unsqueeze(1)
    pos = pos.unsqueeze(1)
    
    pos_sim = self.cos(anchor, pos)
    neg_sim = self.cos(anchor, neg)
    
    loss = torch.clamp(self.margin - pos_sim + neg_sim, min=0)
    return loss.mean()