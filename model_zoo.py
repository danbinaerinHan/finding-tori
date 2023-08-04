import torch.nn as nn
from pitch_utils import filter_by_confidence
from modules import ConvNorm, ContextAttention
from model_utils import cal_conv_parameters


class CnnEncoder(nn.Module):
  def __init__(self, hparams):
    super(CnnEncoder, self).__init__()
    self.hidden_size = hparams.hidden_size
    self.input_size = hparams.input_size
    self.kernel_size = hparams.kernel_size
    self.num_layers = hparams.num_layers
    self.embed_size = hparams.embed_size

    if hparams.use_pre_encoder:
      self.use_pre_encoder = True
      self.pre_encoder = nn.Linear(self.input_size, self.hidden_size)
      self.cnn_input_size = self.hidden_size
    else:
      self.use_pre_encoder = False
      self.cnn_input_size = self.input_size

    self.encoder = nn.Sequential()
    parameters = cal_conv_parameters(hparams, self.cnn_input_size)

    for i, param in enumerate(parameters):
      self.encoder.add_module(f'conv_{i}', ConvNorm(param['input_channel'], param['output_channel'], self.kernel_size, self.kernel_size//2))
      if param['max_pool'] > 1:
        self.encoder.add_module(f'pool_{i}', nn.MaxPool1d(param['max_pool']))
    self.fc = nn.Linear(parameters[-1]['output_channel'], hparams.embed_size)

    if hparams.summ_type == 'context_attention':
      self.final_attention = ContextAttention(parameters[-1]['output_channel'], num_head=hparams.num_head)
    elif hparams.summ_type == 'max':
      self.final_attention = nn.AdaptiveMaxPool1d(1)
  
  def forward(self, batch):
    out = filter_by_confidence(batch) / 12
    out = self.encoder(out)
    if isinstance(self.final_attention, ContextAttention):
      out = self.final_attention(out.permute(0,2,1))
    else:
      out = self.final_attention(out).squeeze(-1)
    # out = self.final_attention(out.permute(0,2,1))
    out = self.fc(out)
    return out

class CnnClassifier(nn.Module):
  def __init__(self, hparams):
    super(CnnClassifier, self).__init__()
    self.encoder = CnnEncoder(hparams)
    self.classifier = nn.Linear(hparams.embed_size, hparams.num_classes)

  def forward(self, batch):
    out = self.encoder(batch)
    out = self.classifier(out)
    return out