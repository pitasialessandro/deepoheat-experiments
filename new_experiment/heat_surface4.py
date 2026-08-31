"""Training DeepOHeat con k in input (schema k_trainable): due materiali lungo y, (k1, k2, yi)
campionati a ogni batch, loss FD conservativa con k per campione (media armonica sulle facce).

Eredita da heat_surface3.py tutto cio' che ha funzionato: loss FD + bilancio energetico, pin_level
(valido per ogni k: la media del fondo e' a + b*<f> per bilancio energetico, indipendente da k),
augmentation di ampiezza, boundary-layer check. La strada PINN non e' portata avanti (30% vs 1.3%).

Test set: file .npz con chiavi f (N,441), u (N,101,101,51), k1, k2, yi (N,) generati da fd_solver
(make_k_testsets.py). Eseguire dalla root: python new_experiment/heat_surface4.py ...
"""
import os, sys, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import jax, jax.numpy as jnp
import equinox as eqx
import optax
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from models_k import DeepOHeat_k, k_profile_21


def pin_level(u, fc, p):
    """Livello analitico: T media del fondo = a + b*<f> (bilancio energetico, non dipende da k)."""
    if not p.get('pin_level', False):
        return u
    target = p['bottom_a'] + p['bottom_b'] * jnp.mean(fc.reshape(fc.shape[0], -1), axis=1)
    shift = target - jnp.mean(u[:, :, :, 0], axis=(1, 2))
    return u + shift[:, None, None, None]


def make_loss_fd(p, nx, nz):
    """Come make_loss_fd di heat_surface3, ma k per campione: kc (B,ny) dal gradino esatto (k1,k2,yi)
    del batch, media armonica sulle facce y. Lo stencil conservativo gestisce l'interfaccia ovunque sia."""
    a, b = p['bottom_a'], p['bottom_b']
    lam = p['lam']
    xg = jnp.linspace(0, 1, nx).reshape(-1, 1); zg = jnp.linspace(0, 0.5, nz).reshape(-1, 1)
    yg = xg.reshape(-1)
    hx, hz = 1.0 / (nx - 1), 0.5 / (nz - 1)

    def terms(model, fc, k1, k2, yi):
        f = jax.image.resize(fc.reshape(-1, 21, 21), (fc.shape[0], nx, nx), 'linear')[..., None]
        # interfaccia sub-griglia: theta = frazione di segmento/cella nel materiale 1.
        # Facce y (flusso perpendicolare): media armonica pesata; nodi (flussi x/z e BC): media di volume.
        th_f = jnp.clip((yi[:, None] - yg[None, :-1]) / hx, 0.0, 1.0)                # (B,ny-1)
        kf = 1.0 / (th_f / k1[:, None] + (1 - th_f) / k2[:, None])
        lo, hi = jnp.clip(yg - hx / 2, 0, 1), jnp.clip(yg + hx / 2, 0, 1)
        th_c = jnp.clip((yi[:, None] - lo[None, :]) / (hi - lo)[None, :], 0.0, 1.0)  # (B,ny)
        kc = th_c * k1[:, None] + (1 - th_c) * k2[:, None]
        kp = k_profile_21(k1, k2, yi)                                                # input del branch k
        u = pin_level(model(((xg, xg, zg), fc, kp))[..., 0], fc, p)                  # (B,nx,ny,nz)
        Fx = kc[:, None, :, None] * (u[:, 1:] - u[:, :-1]) / hx
        Fy = kf[:, None, :, None] * (u[:, :, 1:] - u[:, :, :-1]) / hx
        Fz = kc[:, None, :, None] * (u[:, :, :, 1:] - u[:, :, :, :-1]) / hz
        div = ((Fx[:, 1:] - Fx[:, :-1]) / hx)[:, :, 1:-1, 1:-1] \
            + ((Fy[:, :, 1:] - Fy[:, :, :-1]) / hx)[:, 1:-1, :, 1:-1] \
            + ((Fz[:, :, :, 1:] - Fz[:, :, :, :-1]) / hz)[:, 1:-1, 1:-1, :]
        pde = jnp.mean(div ** 2)
        uz_top = (3 * u[..., -1] - 4 * u[..., -2] + u[..., -3]) / (2 * hz)
        uz_bot = (-3 * u[..., 0] + 4 * u[..., 1] - u[..., 2]) / (2 * hz)
        top = jnp.mean((kc[:, None, :] * uz_top - f[..., 0]) ** 2)
        bottom = jnp.mean((u[..., 0] - a - b * kc[:, None, :] * uz_bot) ** 2)
        d1 = lambda v0, v1, v2, h: (-3 * v0 + 4 * v1 - v2) / (2 * h)
        side = (jnp.mean(d1(u[:, 0], u[:, 1], u[:, 2], hx) ** 2) + jnp.mean(d1(u[:, -1], u[:, -2], u[:, -3], hx) ** 2)
                + jnp.mean(d1(u[:, :, 0], u[:, :, 1], u[:, :, 2], hx) ** 2) + jnp.mean(d1(u[:, :, -1], u[:, :, -2], u[:, :, -3], hx) ** 2))
        energy = jnp.mean((jnp.mean(kc[:, None, :] * uz_top, axis=(1, 2)) - jnp.mean(kc[:, None, :] * uz_bot, axis=(1, 2))) ** 2)
        total = lam['pde'] * pde + lam['top'] * top + lam['bottom'] * bottom + lam['side'] * side + lam['energy'] * energy
        return total, dict(pde=pde, top=top, bottom=bottom, side=side, energy=energy)

    @eqx.filter_jit
    def loss_and_grad(model, fc, k1, k2, yi):
        (loss, aux), grads = eqx.filter_value_and_grad(terms, has_aux=True)(model, fc, k1, k2, yi)
        return loss, aux, grads
    return loss_and_grad


@eqx.filter_jit
def update(grads, optimizer, opt_state, model):
    updates, opt_state = optimizer.update(grads, opt_state, model)
    return eqx.apply_updates(model, updates), opt_state


def evaluate(model, test, result_dir, p, tag):
    """test: dict con f (N,441), u (N,101,101,51), k1,k2,yi (N,)."""
    x = jnp.linspace(0, 1, 101).reshape(-1, 1); z = jnp.linspace(0, 0.5, 51).reshape(-1, 1)
    fs = jnp.asarray(test['f']); kp = k_profile_21(jnp.asarray(test['k1']), jnp.asarray(test['k2']), jnp.asarray(test['yi']))
    pred = np.asarray(pin_level(model(((x, x, z), fs, kp))[..., 0], fs, p))
    true = np.asarray(test['u'])
    np.save(os.path.join(result_dir, 'u_pred.npy'), pred)
    rel = [np.linalg.norm(pred[i] - true[i]) / np.linalg.norm(true[i]) for i in range(len(true))]
    peak = [25 * (pred[i].max() - true[i].max()) for i in range(len(true))]
    maxerr = [25 * np.abs(pred[i] - true[i]).max() for i in range(len(true))]
    dz = 0.5 / 50
    uzz = np.gradient(np.gradient(pred[0], dz, axis=2), dz, axis=2)
    bl_ratio = float(np.abs(uzz[:, :, 46:]).mean() / (np.abs(uzz[:, :, 5:46]).mean() + 1e-12))
    with open(os.path.join(result_dir, 'eval.txt'), 'w') as fh:
        fh.write(f'boundary-layer check |uzz| top(0.45-0.5)/interno: {bl_ratio:.1f}  (>10 = il modello bara tra i punti)\n')
        for i in range(len(true)):
            fh.write(f"case {i} (k1={test['k1'][i]:.2f} k2={test['k2'][i]:.2f} yi={test['yi'][i]:.2f}): "
                     f'rel_l2 {rel[i]:.4%}  peak_err {peak[i]:+.3f} C  max_abs_err {maxerr[i]:.3f} C\n')
        fh.write(f'mean rel_l2 {np.mean(rel):.4%}  mean |peak_err| {np.mean(np.abs(peak)):.3f} C  mean max_err {np.mean(maxerr):.3f} C\n')
    print(open(os.path.join(result_dir, 'eval.txt')).read())
    i0 = int(np.argmax([np.abs(25 * (pred[i] - true[i])).max() for i in range(len(true))]))  # caso peggiore in figura
    T, P = 25 * true[i0] + 20, 25 * pred[i0] + 20
    kw = dict(extent=[0, 1, 0, 1], origin='lower')
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.4), constrained_layout=True)
    vmin, vmax = T.min(), T.max()
    im = ax[0].imshow(T[:, :, -1].T, cmap='magma', vmin=vmin, vmax=vmax, **kw); ax[0].set_title(f'T vera, top [°C] (caso {i0})'); fig.colorbar(im, ax=ax[0])
    im = ax[1].imshow(P[:, :, -1].T, cmap='magma', vmin=vmin, vmax=vmax, **kw); ax[1].set_title(f'T predetta, top  (rel L2 {rel[i0]:.2%})'); fig.colorbar(im, ax=ax[1])
    e = (P - T)[:, :, -1]; m = np.abs(e).max()
    im = ax[2].imshow(e.T, cmap='RdBu_r', vmin=-m, vmax=m, **kw); ax[2].set_title('errore [°C]'); fig.colorbar(im, ax=ax[2])
    ix = int(np.argmax(T[:, :, -1].max(axis=1)))
    yy = np.linspace(0, 1, 101)
    ax[3].plot(yy, T[ix, :, -1], 'k', lw=2, label='vera'); ax[3].plot(yy, P[ix, :, -1], 'r--', lw=2, label='predetta')
    ax[3].axvline(float(test['yi'][i0]), color='gray', ls=':', label=f"interfaccia y={float(test['yi'][i0]):.2f}")
    ax[3].set_title(f'profilo lungo y in superficie, x={ix/100:.2f}'); ax[3].set_xlabel('y'); ax[3].set_ylabel('T [°C]'); ax[3].legend()
    fig.suptitle(tag); fig.savefig(os.path.join(result_dir, 'eval_worst.png'), dpi=120); plt.close(fig)
    return rel, peak


def sample_k(key, batch, args):
    """Campiona (k1,k2,yi) per mappa secondo --k_mode."""
    if args.k_mode == 'fixed':
        ones = jnp.ones(batch)
        return args.k1 * ones, args.k2 * ones, args.y_interface * ones
    kk1, kk2, kyi = jax.random.split(key, 3)
    yi = jax.random.uniform(kyi, (batch,), minval=args.yi_min, maxval=args.yi_max)
    if args.k_mode == 'yi':
        return args.k1 * jnp.ones(batch), args.k2 * jnp.ones(batch), yi
    lk = (jnp.log(args.k_min), jnp.log(args.k_max))
    k1 = jnp.exp(jax.random.uniform(kk1, (batch,), minval=lk[0], maxval=lk[1]))
    k2 = jnp.exp(jax.random.uniform(kk2, (batch,), minval=lk[0], maxval=lk[1]))
    return k1, k2, yi


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--k_mode', default='full', choices=['fixed', 'yi', 'full'],
                    help='fixed = k costante (sanity); yi = solo interfaccia variabile; full = anche k1,k2')
    ap.add_argument('--k1', type=float, default=1.4); ap.add_argument('--k2', type=float, default=0.5)
    ap.add_argument('--y_interface', type=float, default=0.48)
    ap.add_argument('--k_min', type=float, default=0.3); ap.add_argument('--k_max', type=float, default=2.0)
    ap.add_argument('--yi_min', type=float, default=0.2); ap.add_argument('--yi_max', type=float, default=0.8)
    ap.add_argument('--kink_centers', type=int, default=17, help='0 = nessun dizionario di kink (ablation); M>0 = centri equispaziati in [yi_min, yi_max]')
    ap.add_argument('--bottom_a', type=float, default=0.2); ap.add_argument('--bottom_b', type=float, default=2.0)
    for t in ['pde', 'top', 'bottom', 'side']:
        ap.add_argument(f'--lam_{t}', type=float, default=1.0)
    ap.add_argument('--lam_energy', type=float, default=1.0)
    ap.add_argument('--fd_nx', type=int, default=41); ap.add_argument('--fd_nz', type=int, default=21)
    ap.add_argument('--pin_level', type=int, default=1)
    ap.add_argument('--amp_aug_min', type=float, default=0.1)
    ap.add_argument('--train_data', default='data/fs_train_surface_mixed_p4.npy')
    ap.add_argument('--test', default='data/ktest_ansys11.npz,data/ktest_ansys14.npz,data/ktest_var.npz',
                    help='file .npz separati da virgola (f, u, k1, k2, yi)')
    ap.add_argument('--epochs', type=int, default=30000); ap.add_argument('--decay_steps', type=int, default=600)
    ap.add_argument('--lr', type=float, default=1e-3); ap.add_argument('--batch', type=int, default=50)
    ap.add_argument('--log_epoch', type=int, default=500)
    ap.add_argument('--seed', type=int, default=42); ap.add_argument('--tag', default='')
    ap.add_argument('--init_from', default='', help='model.eqx da cui partire (stessa architettura): fine-tune')
    ap.add_argument('--branch_depth', type=int, default=8); ap.add_argument('--branch_hidden', type=int, default=256)
    ap.add_argument('--kbranch_depth', type=int, default=4); ap.add_argument('--kbranch_hidden', type=int, default=64)
    ap.add_argument('--trunk_depth', type=int, default=3); ap.add_argument('--trunk_hidden', type=int, default=64)
    ap.add_argument('--r', type=int, default=128)
    args = ap.parse_args()

    p = dict(bottom_a=args.bottom_a, bottom_b=args.bottom_b, pin_level=bool(args.pin_level),
             lam=dict(pde=args.lam_pde, top=args.lam_top, bottom=args.lam_bottom, side=args.lam_side, energy=args.lam_energy))
    fs_train = jnp.asarray(np.load(args.train_data).reshape(-1, 441).astype(np.float32))
    tests = [(dict(np.load(f)), os.path.basename(f).replace('.npz', '')) for f in args.test.split(',') if f]

    tag = f'k{args.k_mode}{args.tag}'
    result_dir = os.path.join('results', 'results_ktrain', tag); os.makedirs(result_dir, exist_ok=True)
    with open(os.path.join(result_dir, 'args.txt'), 'w') as fh:
        fh.write('\n'.join(f'{k}={v}' for k, v in vars(args).items()))

    centers = tuple(np.linspace(args.yi_min, args.yi_max, args.kink_centers)) if args.kink_centers > 0 else ()
    key = jax.random.PRNGKey(args.seed); key, sub = jax.random.split(key)
    model = DeepOHeat_k(441, kink_centers=centers,
                        branch_depth=args.branch_depth, branch_hidden=args.branch_hidden,
                        kbranch_depth=args.kbranch_depth, kbranch_hidden=args.kbranch_hidden,
                        trunk_depth=args.trunk_depth, trunk_hidden=args.trunk_hidden, rank=args.r, key=sub)
    if args.init_from:
        model = eqx.tree_deserialise_leaves(args.init_from, model)
    optimizer = optax.adam(optax.exponential_decay(args.lr, args.decay_steps, 0.9))
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    loss_and_grad = make_loss_fd(p, args.fd_nx, args.fd_nz)

    log = open(os.path.join(result_dir, 'loss_terms.csv'), 'w'); log.write('epoch,total,pde,top,bottom,side,energy\n')
    t0 = time.time()
    for epoch in range(args.epochs):
        key, sub, subk = jax.random.split(key, 3)
        fc = fs_train[jax.random.choice(sub, fs_train.shape[0], (args.batch,), replace=False)]
        if args.amp_aug_min > 0:
            key, ka, kb = jax.random.split(key, 3)
            scale = jnp.exp(jax.random.uniform(ka, (args.batch, 1), minval=jnp.log(args.amp_aug_min), maxval=0.0))
            scale = jnp.where(jax.random.bernoulli(kb, 0.5, (args.batch, 1)), scale, 1.0)
            fc = fc * scale
        k1, k2, yi = sample_k(subk, args.batch, args)
        loss, aux, grads = loss_and_grad(model, fc, k1, k2, yi)
        model, opt_state = update(grads, optimizer, opt_state, model)
        if epoch % args.log_epoch == 0 or epoch == args.epochs - 1:
            a = {k: float(v) for k, v in aux.items()}
            print(f"Epoch {epoch+1}/{args.epochs} total {float(loss):.3e} | pde {a['pde']:.2e} top {a['top']:.2e} "
                  f"bottom {a['bottom']:.2e} side {a['side']:.2e} energy {a['energy']:.2e}", flush=True)
            log.write(f"{epoch+1},{float(loss)},{a['pde']},{a['top']},{a['bottom']},{a['side']},{a['energy']}\n"); log.flush()
    log.close()
    runtime = time.time() - t0
    print(f'Runtime {runtime:.1f}s ({1000*runtime/args.epochs:.2f} ms/iter)')
    eqx.tree_serialise_leaves(os.path.join(result_dir, 'model.eqx'), model)
    for i, (test, name) in enumerate(tests):
        d = result_dir if i == 0 else os.path.join(result_dir, f'eval_{name}')
        os.makedirs(d, exist_ok=True)
        print(f'--- test {name} ---')
        evaluate(model, test, d, p, f'{tag} — {name}')
