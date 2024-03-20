import torch 
from torch import nn 
from torch.nn import functional as F

from einops import rearrange

from .net import build_model
from .vit import Block
from .iresnet import iresnet100
from imports import *


class FaceFeatureModel(nn.Module):
  def __init__(self, cfg):
    super().__init__()

    self.cfg = cfg
    bname = cfg.TRAINING.MODEL
    if bname == "ELASTIC":
      self.model = iresnet100(num_features = cfg.TRAINING.FEATS)
      statedict = torch.load('./checkpoints/295672backbone.pth')
    
    elif bname == "ADA":
      self.model = build_model('ir_101')
      statedict = torch.load('./checkpoints/adaface_ir101_webface12m.ckpt')['state_dict']
      statedict = {key[6:]:val for key, val in statedict.items() if key.startswith('model.')}

    self.model.load_state_dict(statedict)

    self.get_token_probs = nn.Sequential(
                              Block(dim=512, num_heads=1, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                              drop=0.1, attn_drop=0.00, drop_path=0.00, norm_layer=nn.LayerNorm,
                              init_values=0.),
                              nn.Linear(512, 1),
                              torch.nn.Flatten(start_dim=1),
                              )
  
  # def forward(self, x):
  #   x_gallery, x_probe = x.chunk(chunks=2, dim=2)
    
  #   x_gallery = rearrange(x_gallery, 'b c t h w -> (b t) c h w')
  #   x_probe = rearrange(x_probe, 'b c t h w -> (b t) c h w')
    
  #   z_gallery, zg_norm = self.model(x_gallery)
  #   z_probe, zf_norm = self.model(x_probe)
  #   return (z_gallery, zg_norm), (z_probe, zf_norm)

  def forward(self, x):
    b, c, t, h, w = x.size()
    x_gallery, x_probe = x.chunk(chunks=2, dim=2)
    
    x_gallery = rearrange(x_gallery, 'b c t h w -> (b t) c h w')
    x_probe = rearrange(x_probe, 'b c t h w -> (b t) c h w')
    
    if self.cfg.TRAINING.MODEL == "ELASTIC":
      z_gallery = self.model(x_gallery)
      z_probe = self.model(x_probe)
    elif self.cfg.TRAINING.MODEL == "ADA":
      z_gallery, _ = self.model(x_gallery)
      z_probe, _ = self.model(x_probe)
    

    _, d = z_gallery.size()
    z_gallery = rearrange(z_gallery, '(b t) d -> b t d', b=b, t=t//2, d=d)
    z_probe = rearrange(z_probe, '(b t) d -> b t d', b=b, t=t//2, d=d)

    p_x_gallery = F.softmax(self.get_token_probs(z_gallery), dim=-1)
    p_x_gallery = torch.nan_to_num(p_x_gallery)
    p_x_gallery = p_x_gallery.view(b, t//2, 1)

    p_x_probe = F.softmax(self.get_token_probs(z_probe), dim=-1)
    p_x_probe = torch.nan_to_num(p_x_probe)
    p_x_probe = p_x_probe.view(b, t//2, 1)

    weighted_emb_gallery = F.normalize((z_gallery * p_x_gallery).sum(dim=1), dim=-1)
    weighted_emb_probe = F.normalize((z_probe * p_x_probe).sum(dim=1), dim=-1)


    return weighted_emb_gallery, weighted_emb_probe