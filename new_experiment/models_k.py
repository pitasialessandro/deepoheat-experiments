"""DeepOHeat con conduttivita' k come input (schema k_trainable, Fig. 1): un secondo branch riceve
il profilo k(y) campionato sui 21 punti sensore e il suo output moltiplica (Hadamard) quello del
branch della mappa di potenza; il trunk resta separabile.

Il gomito all'interfaccia non puo' piu' essere una feature fissa |y - 0.48|: con yi variabile il trunk y
riceve un dizionario di kink [y, |y-c_1|, ..., |y-c_M|] su una griglia fissa di centri; e' il branch k,
pesando le componenti di rango, a piazzare il gomito dove serve.
"""
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import equinox.nn as nn
from kan import ChebyKAN


def identity(x):
    return x


def k_profile_21(k1, k2, yi, n=21):
    """Profilo k sui punti sensore: media di k sulla cella del sensore (frazione di volume),
    cosi' la posizione dell'interfaccia e' ricostruibile esattamente dall'input anche tra i sensori.
    k1,k2,yi: scalari o array (B,). Ritorna (..., n)."""
    s = jnp.linspace(0.0, 1.0, n)
    h = 1.0 / (n - 1)
    lo, hi = jnp.clip(s - h / 2, 0.0, 1.0), jnp.clip(s + h / 2, 0.0, 1.0)
    k1, k2, yi = jnp.atleast_1d(k1)[:, None], jnp.atleast_1d(k2)[:, None], jnp.atleast_1d(yi)[:, None]
    frac = jnp.clip((yi - lo) / (hi - lo), 0.0, 1.0)      # frazione di cella nel materiale 1
    return frac * k1 + (1 - frac) * k2


class DeepOHeat_k(eqx.Module):
    trunk_x: eqx.Module
    trunk_y: eqx.Module
    trunk_z: eqx.Module
    branch_f: eqx.Module
    branch_k: eqx.Module
    field_dim: int = eqx.field(static=True)
    rank: int = eqx.field(static=True)
    kink_centers: tuple = eqx.field(static=True)          # centri fissi (non allenati)

    def __init__(self, branch_dim, k_dim=21, kink_centers=(), field_dim=1, branch_depth=8, branch_hidden=256,
                 kbranch_depth=4, kbranch_hidden=64, trunk_depth=3, trunk_hidden=64, rank=128, key=None):
        keys = jax.random.split(key, 5)
        mk = lambda in_size, k: eqx.filter_vmap(ChebyKAN(in_size=in_size, out_size=rank * field_dim,
                                                          width_size=trunk_hidden, depth=trunk_depth, key=k))
        self.kink_centers = tuple(float(c) for c in kink_centers)
        self.trunk_x = mk(1, keys[0])
        self.trunk_y = mk(1 + len(self.kink_centers), keys[1])
        self.trunk_z = mk(1, keys[2])
        self.branch_f = eqx.filter_vmap(nn.MLP(branch_dim, rank * field_dim, branch_hidden, branch_depth,
                                               activation=jax.nn.swish, final_activation=identity, key=keys[3]))
        self.branch_k = eqx.filter_vmap(nn.MLP(k_dim, rank * field_dim, kbranch_hidden, kbranch_depth,
                                               activation=jax.nn.swish, final_activation=identity, key=keys[4]))
        self.field_dim, self.rank = field_dim, rank

    def __call__(self, x__f__k):
        (x, y, z), f, kp = x__f__k
        if f.ndim == 1:
            f = f[None, :]
        if kp.ndim == 1:
            kp = kp[None, :]
        if self.kink_centers:
            c = jnp.asarray(self.kink_centers)
            y_in = jnp.concatenate([y, jnp.abs(y - c[None, :])], axis=-1)
        else:
            y_in = y
        tx = self.trunk_x(x).reshape(x.shape[0], self.field_dim, self.rank)
        ty = self.trunk_y(y_in).reshape(y.shape[0], self.field_dim, self.rank)
        tz = self.trunk_z(z).reshape(z.shape[0], self.field_dim, self.rank)
        bf = self.branch_f(f).reshape(-1, self.field_dim, self.rank)
        bk = self.branch_k(jnp.log(kp)).reshape(-1, self.field_dim, self.rank)
        b = bf * bk                                        # prodotto di Hadamard (Fig. 1 del PDF)
        return jnp.einsum('ifr,jfr,kfr,bfr->bijkf', tx, ty, tz, b, optimize='optimal')
