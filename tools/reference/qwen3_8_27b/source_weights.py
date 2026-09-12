"""Independent logical Text views of the local original BF16 safetensors.

Diagnostic only: no converter recipes or production binder are called. Q/gate
rows are explicitly deinterleaved by head, convolution is time-major, and packed
MLP/GDN views follow the model formulas. Source tensors are never requantized.
"""
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open


class SourceWeights:
    def __init__(self, path):
        self.path = Path(path).resolve(strict=True)
        self.index = json.loads((self.path/'model.safetensors.index.json').read_text())['weight_map']
        config = json.loads((self.path/'config.json').read_text())['text_config']
        for field,value in dict(hidden_size=5120,num_hidden_layers=64,
                                num_attention_heads=24,num_key_value_heads=4,
                                head_dim=256,intermediate_size=17408).items():
            if config[field] != value:
                raise ValueError(f'unexpected source geometry: {field}')

    def read(self, name, rows=None):
        return self.read_tensor(name,rows).float().numpy()

    def read_tensor(self, name, rows=None):
        """Read-only mapped source tensor; retain BF16 until the explicit GPU cast."""
        with safe_open(self.path/self.index[name],framework='pt',device='cpu') as f:
            return f.get_tensor(name) if rows is None else f.get_slice(name)[rows]

    def tensor(self, name):
        return self.matrix(name,dtype=np.float32)

    def matrix(self, name, start=0, count=None, *, dtype=np.float64):
        return self.logical_tensor(name,start,count).float().numpy().astype(dtype,copy=False)

    def logical_tensor(self, name, start=0, count=None):
        endpoint = {'text/token_embedding':'model.language_model.embed_tokens.weight',
                    'text/output_head':'lm_head.weight',
                    'text/final_norm':'model.language_model.norm.weight'}
        if name in endpoint:
            rows = slice(start,None if count is None else start+count)
            return self.read_tensor(endpoint[name],rows)
        _,_,layer,role = name.split('/',3)
        prefix = f'model.language_model.layers.{int(layer)}.'
        def read(suffix):
            return self.read_tensor(prefix+suffix)
        direct = {
            'input_norm':'input_layernorm.weight',
            'post_attention_norm':'post_attention_layernorm.weight',
            'attention/query_norm':'self_attn.q_norm.weight',
            'attention/key_norm':'self_attn.k_norm.weight',
            'attention/output':'self_attn.o_proj.weight',
            'gdn/a_log':'linear_attn.A_log',
            'gdn/dt_bias':'linear_attn.dt_bias',
            'gdn/a_projection':'linear_attn.in_proj_a.weight',
            'gdn/b_projection':'linear_attn.in_proj_b.weight',
            'gdn/norm':'linear_attn.norm.weight',
            'gdn/output':'linear_attn.out_proj.weight',
            'mlp/down':'mlp.down_proj.weight',
        }
        if role in direct:
            result = read(direct[role])
        elif role == 'attention/query_key_gate_value':
            qg = read('self_attn.q_proj.weight').reshape(24,512,5120)
            result = torch.cat((qg[:,:256].reshape(6144,5120),
                                     read('self_attn.k_proj.weight'),
                                     qg[:,256:].reshape(6144,5120),
                                     read('self_attn.v_proj.weight')))
        elif role == 'gdn/query_key_value_z':
            result = torch.cat((read('linear_attn.in_proj_qkv.weight'),
                                     read('linear_attn.in_proj_z.weight')))
        elif role == 'gdn/a_b_projection':
            result = torch.cat((read('linear_attn.in_proj_a.weight'),read('linear_attn.in_proj_b.weight')))
        elif role == 'gdn/convolution':
            result = read('linear_attn.conv1d.weight')[:,0,:].T.contiguous()
        elif role == 'mlp/gate_up':
            result = torch.cat((read('mlp.gate_proj.weight'),read('mlp.up_proj.weight')))
        else:
            raise ValueError(f'not a source Text weight: {name}')
        return result[start:None if count is None else start+count]

    def close(self):
        pass
