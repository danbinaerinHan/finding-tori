import numpy as np
import torch
import torch.nn as nn


class ContextAttention(nn.Module):
  def __init__(self, size, num_head):
    super(ContextAttention, self).__init__()
    self.attention_net = nn.Linear(size, size)
    self.num_head = num_head

    if size % num_head != 0:
        raise ValueError("size must be dividable by num_head", size, num_head)
    self.head_size = int(size/num_head)
    self.context_vector = torch.nn.Parameter(torch.Tensor(num_head, self.head_size, 1))
    nn.init.uniform_(self.context_vector, a=-1, b=1)

  def get_attention(self, x):
    attention = self.attention_net(x)
    attention_tanh = torch.tanh(attention)
    attention_split = torch.stack(attention_tanh.split(split_size=self.head_size, dim=2), dim=0)
    similarity = torch.bmm(attention_split.view(self.num_head, -1, self.head_size), self.context_vector)
    similarity = similarity.view(self.num_head, x.shape[0], -1).permute(1,2,0)
    return similarity

  def forward(self, x):
    attention = self.attention_net(x)
    attention_tanh = torch.tanh(attention)
    if self.head_size != 1:
      attention_split = torch.stack(attention_tanh.split(split_size=self.head_size, dim=2), dim=0)
      similarity = torch.bmm(attention_split.view(self.num_head, -1, self.head_size), self.context_vector)
      similarity = similarity.view(self.num_head, x.shape[0], -1).permute(1,2,0)
      similarity[x.sum(-1)==0] = -1e10 # mask out zero padded_ones
      softmax_weight = torch.softmax(similarity, dim=1)

      x_split = torch.stack(x.split(split_size=self.head_size, dim=2), dim=2)
      weighted_x = x_split * softmax_weight.unsqueeze(-1).repeat(1,1,1, x_split.shape[-1])
      attention = weighted_x.view(x_split.shape[0], x_split.shape[1], x.shape[-1])
      

    else:
      softmax_weight = torch.softmax(attention, dim=1)
      attention = softmax_weight * x

    sum_attention = torch.sum(attention, dim=1)
    return sum_attention

class ConvNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super().__init__()
        self.conv_norm = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv_norm(x)



# Modules for harmonic filters
def hz_to_midi(hz):
    return 12 * (torch.log2(hz) - np.log2(440.0)) + 69

def midi_to_hz(midi):
    return 440.0 * (2.0 ** ((midi - 69.0)/12.0))
