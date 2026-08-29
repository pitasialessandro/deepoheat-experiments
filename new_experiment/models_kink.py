"""DeepOHeat_v1 con trunk in y che riceve [y, |y - y_i|]: la feature |y - y_i| e' continua con derivata
discontinua, la stessa forma della soluzione a due materiali. Nella struttura separabile basta darla al
trunk in y per rappresentare il kink su tutto il piano dell'interfaccia."""
import jax
import jax.numpy as jnp
import equinox as eqx
import equinox.nn as nn
from kan import ChebyKAN


def identity(x):
    return x


class DeepOHeat_kink(eqx.Module):
    trunk_x: eqx.Module
    trunk_y: eqx.Module
    trunk_z: eqx.Module
    branch: eqx.Module
    field_dim: int = eqx.field(static=True)
    rank: int = eqx.field(static=True)
    y_interface: float = eqx.field(static=True)
    use_kink: bool = eqx.field(static=True)

    def __init__(self, branch_dim, y_interface=0.48, use_kink=True, field_dim=1, branch_depth=8, branch_hidden=256,
                 trunk_depth=3, trunk_hidden=64, rank=128, key=None):
        keys = jax.random.split(key, 4)
        mk = lambda in_size, k: eqx.filter_vmap(ChebyKAN(in_size=in_size, out_size=rank * field_dim,
                                                          width_size=trunk_hidden, depth=trunk_depth, key=k))
        self.trunk_x = mk(1, keys[0])
        self.trunk_y = mk(2 if use_kink else 1, keys[1])
        self.trunk_z = mk(1, keys[2])
        self.branch = eqx.filter_vmap(nn.MLP(branch_dim, rank * field_dim, branch_hidden, branch_depth,
                                             activation=jax.nn.swish, final_activation=identity, key=keys[3]))
        self.field_dim, self.rank = field_dim, rank
        self.y_interface, self.use_kink = float(y_interface), bool(use_kink)

    def __call__(self, x__f):
        (x, y, z), f = x__f
        if f.ndim == 1:
            f = f[None, :]
        y_in = jnp.concatenate([y, jnp.abs(y - self.y_interface)], axis=-1) if self.use_kink else y
        tx = self.trunk_x(x).reshape(x.shape[0], self.field_dim, self.rank)
        ty = self.trunk_y(y_in).reshape(y.shape[0], self.field_dim, self.rank)
        tz = self.trunk_z(z).reshape(z.shape[0], self.field_dim, self.rank)
        b = self.branch(f).reshape(-1, self.field_dim, self.rank)
        return jnp.einsum('ifr,jfr,kfr,bfr->bijkf', tx, ty, tz, b, optimize='optimal')
