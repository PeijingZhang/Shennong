import torch
import torch.nn as nn
import numpy as np

from typing import Optional

from ..trvae._utils import one_hot_encoder

class MaskedLinear(nn.Linear):
    def __init__(self, n_in, n_out, mask, bias=True):
        # mask should have the same dimensions as the transposed linear weight
        # n_input x n_output_nodes
        if mask is not None:
            print(f"MaskedLinear init - mask shape: {mask.shape}, n_in: {n_in}, n_out: {n_out}")
            if n_in != mask.shape[0] or n_out != mask.shape[1]:
                raise ValueError(f'Incorrect shape of the mask. Expected ({n_in}, {n_out}), got {mask.shape}')

        super().__init__(n_in, n_out, bias)
        
        device = mask.device if mask is not None else self.weight.device
        
        self.weight = nn.Parameter(self.weight.to(device))
        if self.bias is not None:
            self.bias = nn.Parameter(self.bias.to(device))
            
        if mask is not None:
            mask = mask.to(device)
            mask = mask.t()
            self.register_buffer('mask', mask)
            
            with torch.no_grad():
                self.weight.data = self.weight.data * self.mask
        else:
            self.register_buffer('mask', None)

    def forward(self, input):
        # print(f"MaskedLinear forward - input shape: {input.shape}, weight shape: {self.weight.shape}")
        
        if input.device != self.weight.device:
            input = input.to(self.weight.device)
            
        weight = self.weight * self.mask if self.mask is not None else self.weight
        
        return nn.functional.linear(input, weight, self.bias)    


class MaskedCondLayers(nn.Module):
    def __init__(
        self,
        n_in: int,
        n_out: int,
        n_cond: int,
        bias: bool,
        n_ext: int = 0,
        n_ext_m: int = 0,
        mask: Optional[torch.Tensor] = None,
        ext_mask: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.n_cond = n_cond
        self.n_ext = n_ext
        self.n_ext_m = n_ext_m
        
        if mask is None:
            self.expr_L = nn.Linear(n_in, n_out, bias=bias)
        else:
            if mask.shape[0] != n_in and mask.shape[1] == n_in:
                mask = mask.t()
            self.expr_L = MaskedLinear(n_in, n_out, mask, bias=bias)

        if self.n_cond != 0:
            self.cond_L = nn.Linear(self.n_cond, n_out, bias=False)

        if self.n_ext != 0:
            self.ext_L = nn.Linear(self.n_ext, n_out, bias=False)

        if self.n_ext_m != 0:
            if ext_mask is not None:
                if ext_mask.shape[0] != self.n_ext_m and ext_mask.shape[1] == self.n_ext_m:
                    ext_mask = ext_mask.t()
                self.ext_L_m = MaskedLinear(self.n_ext_m, n_out, ext_mask, bias=False)
            else:
                self.ext_L_m = nn.Linear(self.n_ext_m, n_out, bias=False)

    def forward(self, x):
        expr = x
        if self.n_cond != 0:
            expr, cond = torch.split(x, [x.shape[1] - self.n_cond, self.n_cond], dim=1)
        else:
            cond = None

        if self.n_ext != 0:
            expr, ext = torch.split(expr, [expr.shape[1] - self.n_ext, self.n_ext], dim=1)
        else:
            ext = None

        if self.n_ext_m != 0:
            expr, ext_m = torch.split(expr, [expr.shape[1] - self.n_ext_m, self.n_ext_m], dim=1)
        else:
            ext_m = None

        out = self.expr_L(expr)
        if ext is not None:
            out = out + self.ext_L(ext)
        if ext_m is not None:
            out = out + self.ext_L_m(ext_m)
        if cond is not None:
            out = out + self.cond_L(cond)
        return out


class MaskedLinearDecoder(nn.Module):
    def __init__(self, in_dim, out_dim, n_cond, mask, ext_mask, recon_loss,
                 last_layer=None, n_ext=0, n_ext_m=0):
        super().__init__()
        device = mask.device if mask is not None else ('cuda' if torch.cuda.is_available() else 'cpu')  ### to(device)

        if recon_loss == "mse":
            if last_layer == "softmax":
                raise ValueError("Can't specify softmax last layer with mse loss.")
            last_layer = "identity" if last_layer is None else last_layer
        elif recon_loss == "nb":
            last_layer = "softmax" if last_layer is None else last_layer
        else:
            raise ValueError("Unrecognized loss.")

        print("Decoder Architecture:")
        print("\tMasked linear layer in, ext_m, ext, cond, out: ", in_dim, n_ext_m, n_ext, n_cond, out_dim)
        if mask is not None:
            print('\twith hard mask.')
        else:
            print('\twith soft mask.')

        self.n_ext = n_ext
        self.n_ext_m = n_ext_m
        self.n_cond = 0 if n_cond is None else n_cond

        self.L0 = MaskedCondLayers(
            in_dim, out_dim, self.n_cond, bias=False,
            n_ext=n_ext, n_ext_m=n_ext_m,
            mask=mask.to(device) if mask is not None else None,
            ext_mask=ext_mask.to(device) if ext_mask is not None else None
        )

        if last_layer == "softmax":
            self.mean_decoder = nn.Softmax(dim=-1)
        elif last_layer == "softplus":
            self.mean_decoder = nn.Softplus()
        elif last_layer == "exp":
            self.mean_decoder = torch.exp
        elif last_layer == "relu":
            self.mean_decoder = nn.ReLU()
        elif last_layer == "identity":
            self.mean_decoder = lambda a: a
        else:
            raise ValueError("Unrecognized last layer.")

        print("Last Decoder layer:", last_layer)
        self.to(device)

    def forward(self, z, batch=None):
        device = next(self.parameters()).device
        z = z.to(device)
        if batch is not None:
            batch = batch.to(device)
            batch = one_hot_encoder(batch, n_cls=self.n_cond)
            z = torch.cat((z, batch), dim=-1)
            
        dec_latent = self.L0(z)
        recon_x = self.mean_decoder(dec_latent)
        
        return recon_x, dec_latent

    def nonzero_terms(self):
        """Return indices of active terms.
        Active terms are the terms which were not deactivated by the group lasso regularization.
        """
        v = self.L0.expr_L.weight.data
        nz = (v.norm(p=1, dim=0) > 0).cpu().numpy()
        nz = np.append(nz, np.full(self.n_ext_m, True))
        nz = np.append(nz, np.full(self.n_ext, True))
        return nz

    def n_inactive_terms(self):
        """Return the number of inactive terms.
        Inactive terms are those that were deactivated by the group lasso regularization.
        """
        n = (~self.nonzero_terms()).sum()
        return int(n)


class ExtEncoder(nn.Module):
    def __init__(self,
                 layer_sizes: list,
                 latent_dim: int,
                 use_bn: bool,
                 use_ln: bool,
                 use_dr: bool,
                 dr_rate: float,
                 num_classes: Optional[int] = None,
                 n_expand: int = 0):
        super().__init__() 
        self.n_classes = 0
        self.n_expand = n_expand
        if num_classes is not None:
            self.n_classes = num_classes
        self.FC = None
        if len(layer_sizes) > 1:
            print("Encoder Architecture:")
            self.FC = nn.Sequential()
            for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
                if i == 0:
                    print("\tInput Layer in, out and cond:", in_size, out_size, self.n_classes)
                    self.FC.add_module(name=f"L{i}", module=MaskedCondLayers(in_size,
                                                                             out_size,
                                                                             self.n_classes,
                                                                             bias=True))
                else:
                    print("\tHidden Layer", i, "in/out:", in_size, out_size)
                    self.FC.add_module(name=f"L{i}", module=nn.Linear(in_size, out_size, bias=True))
                if use_bn:
                    self.FC.add_module(f"N{i}", module=nn.BatchNorm1d(out_size, affine=True))
                elif use_ln:
                    self.FC.add_module(f"N{i}", module=nn.LayerNorm(out_size, elementwise_affine=False))
                self.FC.add_module(name=f"A{i}", module=nn.ReLU())
                if use_dr:
                    self.FC.add_module(name=f"D{i}", module=nn.Dropout(p=dr_rate))
        print("\tMean/Var Layer in/out:", layer_sizes[-1], latent_dim)
        self.mean_encoder = nn.Linear(layer_sizes[-1], latent_dim)
        self.log_var_encoder = nn.Linear(layer_sizes[-1], latent_dim)

        if self.n_expand != 0:
            print("\tExpanded Mean/Var Layer in/out:", layer_sizes[-1], self.n_expand)
            self.expand_mean_encoder = nn.Linear(layer_sizes[-1], self.n_expand)
            self.expand_var_encoder = nn.Linear(layer_sizes[-1], self.n_expand)

    def forward(self, x: torch.Tensor, batch=None):
        """Forward pass through the encoder network.
        
        Args:
            x: Input tensor of shape [batch_size, input_dim]
            batch: Optional batch information for conditioning
            
        Returns:
            means, log_vars: Encoded mean and log variance tensors
        """
        device = next(self.parameters()).device
        x = x.to(device)
        
        if batch is not None:
            batch = one_hot_encoder(batch, n_cls=self.n_classes)
            x = torch.cat((x, batch), dim=-1)
            
        if hasattr(self, 'FC'):
            x = self.FC(x) 
            
        means = self.mean_encoder(x)
        log_vars = self.log_var_encoder(x)

        if self.n_expand != 0:
            means = torch.cat((means, self.expand_mean_encoder(x)), dim=-1)
            log_vars = torch.cat((log_vars, self.expand_var_encoder(x)), dim=-1)
        return means, log_vars
