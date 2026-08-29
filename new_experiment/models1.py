import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import equinox.nn as nn
from kan import ChebyKAN


# =========================================================
# einsum helpers
# =========================================================
c2 = ['i', 'j', 'k', 'l', 'm', 'n', 'o', 'p']
c1 = [c + 'yz' for c in c2]


# =========================================================
# activations
# =========================================================
@jax.jit
def sine(x):
    return jnp.sin(x)


@jax.jit
def identity(x):
    return x


# =========================================================
# DeepOHeat baseline
# =========================================================
class DeepOHeat(eqx.Module):
    dim: int
    branch_dim: int
    field_dim: int
    rank: int
    trunk: eqx.Module
    branch: eqx.Module
    B: jax.Array

    def __init__(
        self,
        dim,
        branch_dim,
        field_dim=1,
        branch_depth=3,
        branch_hidden=64,
        trunk_depth=3,
        trunk_hidden=64,
        rank=64,
        branch_activation=jax.nn.swish,
        branch_final_activation=identity,
        trunk_activation=jax.nn.swish,
        trunk_final_activation=identity,
        key=None,
    ):
        super().__init__()
        key, subkey = jax.random.split(key)
        subkey1, subkey2 = jax.random.split(subkey)

        self.B = 2 * jnp.pi * jax.random.normal(key, shape=(dim, 64))
        self.trunk = eqx.filter_vmap(
            nn.MLP(
                128,
                rank * field_dim,
                trunk_hidden,
                trunk_depth,
                activation=trunk_activation,
                final_activation=trunk_final_activation,
                key=subkey1,
            )
        )
        self.branch = nn.MLP(
            branch_dim,
            rank * field_dim,
            branch_hidden,
            branch_depth,
            activation=branch_activation,
            final_activation=branch_final_activation,
            key=subkey2,
        )

        self.dim = dim
        self.branch_dim = branch_dim
        self.field_dim = field_dim
        self.rank = rank

    def __call__(self, x__f):
        x, f = x__f
        x = jnp.concatenate(x, axis=-1)
        x = jnp.concatenate((jnp.cos(x @ self.B), jnp.sin(x @ self.B)), axis=1)
        t = self.trunk(x).reshape(-1, self.field_dim, self.rank)
        b = self.branch(f).reshape(self.field_dim, self.rank)
        return jnp.einsum("ijk,jk->ij", t, b, optimize="optimal")


# =========================================================
# DeepOHeat + KAN
# =========================================================
class DeepOHeat_KAN(eqx.Module):
    dim: int
    branch_dim: int
    field_dim: int
    rank: int
    trunk: eqx.Module
    branch: eqx.Module

    def __init__(
        self,
        dim,
        branch_dim,
        field_dim=1,
        branch_depth=3,
        branch_hidden=64,
        trunk_depth=3,
        trunk_hidden=64,
        rank=64,
        branch_activation=jax.nn.swish,
        branch_final_activation=identity,
        key=None,
    ):
        super().__init__()

        subkey1, subkey2 = jax.random.split(key)

        self.trunk = eqx.filter_vmap(
            ChebyKAN(
                in_size=dim,
                out_size=rank * field_dim,
                width_size=trunk_hidden,
                depth=trunk_depth,
                key=subkey1,
            )
        )

        self.branch = nn.MLP(
            branch_dim,
            rank * field_dim,
            branch_hidden,
            branch_depth,
            activation=branch_activation,
            final_activation=branch_final_activation,
            key=subkey2,
        )

        self.dim = dim
        self.branch_dim = branch_dim
        self.field_dim = field_dim
        self.rank = rank

    def __call__(self, x__f):
        x, f = x__f
        x = jnp.concatenate(x, axis=-1)
        t = self.trunk(x).reshape(-1, self.field_dim, self.rank)
        b = self.branch(f).reshape(self.field_dim, self.rank)
        return jnp.einsum("ijk,jk->ij", t, b, optimize="optimal")


# =========================================================
# DeepOHeat separable trunk
# =========================================================
class DeepOHeat_ST(eqx.Module):
    dim: int
    branch_dim: int
    field_dim: int
    trunk: list
    branch: eqx.Module
    rank: int
    outer_product_string: str
    B: jax.Array

    def __init__(
        self,
        dim,
        branch_dim,
        field_dim=1,
        branch_depth=3,
        branch_hidden=64,
        trunk_depth=3,
        trunk_hidden=64,
        rank=64,
        branch_activation=jax.nn.swish,
        branch_final_activation=identity,
        trunk_activation=jax.nn.swish,
        trunk_final_activation=identity,
        key=None,
    ):
        super().__init__()

        def make_ensemble(keys):
            mlps = []
            for i in range(len(keys)):
                mlp = eqx.filter_vmap(
                    nn.MLP(
                        128,
                        rank * field_dim,
                        trunk_hidden,
                        trunk_depth,
                        activation=trunk_activation,
                        final_activation=trunk_final_activation,
                        key=keys[i],
                    )
                )
                mlps.append(mlp)
            return mlps

        subkeys = jax.random.split(key, num=dim + 2)
        self.trunk = make_ensemble(subkeys[:-2])
        self.B = 2 * jnp.pi * jax.random.normal(subkeys[-2], shape=(1, 64))

        self.branch = eqx.filter_vmap(
            nn.MLP(
                branch_dim,
                rank * field_dim,
                branch_hidden,
                branch_depth,
                activation=branch_activation,
                final_activation=branch_final_activation,
                key=subkeys[-1],
            )
        )

        self.dim = dim
        self.field_dim = field_dim
        self.branch_dim = branch_dim
        self.rank = rank

        s1 = ""
        s2 = ""
        for i in range(dim):
            s1 += c1[i] + ","
            s2 += c2[i]
        self.outer_product_string = s1 + "byz->b" + s2 + "y"

    def __call__(self, x__f, return_basis=False):
        x, f = x__f
        ts = []
        for i in range(len(x)):
            xi = jnp.concatenate((jnp.cos(x[i] @ self.B), jnp.sin(x[i] @ self.B)), axis=1)
            ts.append(self.trunk[i](xi).reshape(-1, self.field_dim, self.rank))

        b = self.branch(f).reshape(-1, self.field_dim, self.rank)
        out = jnp.einsum(self.outer_product_string, *ts, b, optimize="optimal")

        if return_basis:
            return ts, b, out
        return out


# =========================================================
# DeepOHeat-v1 separable + KAN
# =========================================================
class DeepOHeat_v1(eqx.Module):
    dim: int
    branch_dim: int
    field_dim: int
    trunk: list
    branch: eqx.Module
    rank: int
    outer_product_string: str

    def __init__(
        self,
        dim,
        branch_dim,
        field_dim=1,
        branch_depth=3,
        branch_hidden=64,
        trunk_depth=3,
        trunk_hidden=64,
        rank=64,
        branch_activation=jax.nn.swish,
        branch_final_activation=identity,
        key=None,
    ):
        super().__init__()

        def make_ensemble(keys):
            kans = []
            for i in range(len(keys)):
                kan = eqx.filter_vmap(
                    ChebyKAN(
                        in_size=1,
                        out_size=rank * field_dim,
                        width_size=trunk_hidden,
                        depth=trunk_depth,
                        key=keys[i],
                    )
                )
                kans.append(kan)
            return kans

        subkeys = jax.random.split(key, num=dim + 1)
        self.trunk = make_ensemble(subkeys[:-1])

        self.branch = eqx.filter_vmap(
            nn.MLP(
                branch_dim,
                rank * field_dim,
                branch_hidden,
                branch_depth,
                activation=branch_activation,
                final_activation=branch_final_activation,
                key=subkeys[-1],
            )
        )

        self.dim = dim
        self.field_dim = field_dim
        self.branch_dim = branch_dim
        self.rank = rank

        s1 = ""
        s2 = ""
        for i in range(dim):
            s1 += c1[i] + ","
            s2 += c2[i]
        self.outer_product_string = s1 + "byz->b" + s2 + "y"

    def __call__(self, x__f, return_basis=False):
        x, f = x__f
        ts = []
        for i in range(len(x)):
            ts.append(self.trunk[i](x[i]).reshape(-1, self.field_dim, self.rank))

        b = self.branch(f).reshape(-1, self.field_dim, self.rank)
        out = jnp.einsum(self.outer_product_string, *ts, b, optimize="optimal")

        if return_basis:
            return ts, b, out
        return out


class DeepOHeat_phi(eqx.Module): # DeepOHeat_v1_k con k fissato, aggiustato con \phi-DeepOHeat
    """
    DeepOHeat-phi con:
    - una sola branch net per la power map completa f
    - trunk separabile su x, y, z
    - trunk_y augmentata con phi(y), embedding di regione

    Forward:
        model(((x, y, z), f))

    Input:
        x: (Nx, 1)
        y: (Ny, 1)
        z: (Nz, 1)
        f: (B, 21*21) oppure (21*21,)

    Output:
        out: (B, Nx, Ny, Nz, field_dim)
    """

    dim: int = eqx.field(static=True)
    branch_dim_f: int = eqx.field(static=True)
    field_dim: int = eqx.field(static=True)
    rank: int = eqx.field(static=True)
    phi_latent_dim: int = eqx.field(static=True)
    interface_y: float = eqx.field(static=True)

    trunk_x: eqx.Module
    trunk_y: eqx.Module
    trunk_z: eqx.Module
    branch_f: eqx.Module

    region_embedding: jax.Array

    def __init__(
        self,
        dim,
        branch_dim=21 * 21,
        interface_y=0.5,
        phi_latent_dim=4,
        field_dim=1,
        branch_depth=3,
        branch_hidden=64,
        trunk_depth=3,
        trunk_hidden=64,
        rank=64,
        branch_activation=jax.nn.swish,
        branch_final_activation=identity,
        key=None,
    ):
        super().__init__()

        if key is None:
            key = jax.random.PRNGKey(0)

        keys = jax.random.split(key, 5)

        # =========================
        # trunks separabili
        # =========================
        self.trunk_x = eqx.filter_vmap(
            ChebyKAN(
                in_size=1,
                out_size=rank * field_dim,
                width_size=trunk_hidden,
                depth=trunk_depth,
                key=keys[0],
            )
        )

        self.trunk_y = eqx.filter_vmap(
            ChebyKAN(
                in_size=1 + phi_latent_dim,   # [y, phi(y)]
                out_size=rank * field_dim,
                width_size=trunk_hidden,
                depth=trunk_depth,
                key=keys[1],
            )
        )

        self.trunk_z = eqx.filter_vmap(
            ChebyKAN(
                in_size=1,
                out_size=rank * field_dim,
                width_size=trunk_hidden,
                depth=trunk_depth,
                key=keys[2],
            )
        )

        # =========================
        # branch unica sulla power map completa
        # =========================
        self.branch_f = eqx.filter_vmap(
            nn.MLP(
                branch_dim,
                rank * field_dim,
                branch_hidden,
                branch_depth,
                activation=branch_activation,
                final_activation=branch_final_activation,
                key=keys[3],
            )
        )

        # =========================
        # embedding categoriale di regione
        # phi(y) = tanh(one_hot(y) @ E^T)
        # =========================
        self.region_embedding = 0.1 * jax.random.normal(keys[4], (phi_latent_dim, 2))

        self.dim = dim
        self.branch_dim_f = branch_dim
        self.field_dim = field_dim
        self.rank = rank
        self.phi_latent_dim = phi_latent_dim
        self.interface_y = float(interface_y)

    def _build_phi(self, y):
        """
        Costruisce phi(y) come embedding categoriale della regione.
        y: (Ny, 1)
        return: (Ny, phi_latent_dim)
        """
        y_vec = y.reshape(-1)

        in_region1 = (y_vec < self.interface_y).astype(jnp.float32)
        in_region2 = 1.0 - in_region1
        one_hot = jnp.stack([in_region1, in_region2], axis=-1)   # (Ny, 2)

        phi = jnp.tanh(one_hot @ self.region_embedding.T)        # (Ny, phi_latent_dim)
        return phi

    def __call__(self, x__f, return_basis=False):
        (x, y, z), f = x__f

        x = jnp.asarray(x, dtype=jnp.float32).reshape(-1, 1)
        y = jnp.asarray(y, dtype=jnp.float32).reshape(-1, 1)
        z = jnp.asarray(z, dtype=jnp.float32).reshape(-1, 1)

        f = jnp.asarray(f, dtype=jnp.float32)
        if f.ndim == 1:
            f = f[None, :]

        # opzionale: piccola normalizzazione della power map
        #f = f / jnp.sqrt(self.branch_dim_f)
        # =========================
        # trunks separabili
        # =========================
        tx = self.trunk_x(x).reshape(x.shape[0], self.field_dim, self.rank)

        phi = self._build_phi(y)                         # (Ny, phi_latent_dim)
        y_phi = jnp.concatenate([y, phi], axis=-1)      # (Ny, 1 + phi_latent_dim)
        ty = self.trunk_y(y_phi).reshape(y.shape[0], self.field_dim, self.rank)

        tz = self.trunk_z(z).reshape(z.shape[0], self.field_dim, self.rank)

        # =========================
        # branch unica
        # =========================
        b = self.branch_f(f).reshape(-1, self.field_dim, self.rank)

        # =========================
        # fusione finale branch + trunk
        # out shape: (B, Nx, Ny, Nz, field_dim)
        # =========================
        out = jnp.einsum(
            "ifr,jfr,kfr,bfr->bijkf",
            tx, ty, tz, b,
            optimize="optimal",
        )

        if return_basis:
            return tx, ty, tz, phi, b, out

        return out