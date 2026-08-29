"""Training DeepOHeat su chip a due materiali (k a gradino lungo y), loss coerente con il setup Ansys.

Differenze rispetto a heat_surface2.py:
- BC fondo parametrica u - a = b*k*uz (Ansys: a=0.2, b=2.0), BC top k(y)*uz = f;
- interfaccia: continuita' del flusso k1*uy(yi-eps) = k2*uy(yi+eps), niente laplaciano sull'interfaccia;
- modello con feature |y - yi| nel trunk y (--model kink) o baseline separabile (--model v1).
Eseguire dalla root della repo: python new_experiment/heat_surface3.py ...
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
from hvp import hvp_fwdfwd
from models_kink import DeepOHeat_kink


def make_loss(p):
    k1, k2, yi, a, b, eps = p['k1'], p['k2'], p['y_interface'], p['bottom_a'], p['bottom_b'], p['eps_if']
    lam = p['lam']

    def terms(model, xc, yc, zc, fc):
        nx, ny = xc.shape[0], yc.shape[0]
        ones = lambda t: jnp.ones(t.shape)
        u = pin_level(model(((xc, yc, zc), fc))[..., 0], fc, p)[..., None]
        ux, uxx = hvp_fwdfwd(lambda x_: model(((x_, yc, zc), fc)), (xc,), (ones(xc),), True)
        uy, uyy = hvp_fwdfwd(lambda y_: model(((xc, y_, zc), fc)), (yc,), (ones(yc),), True)
        uz, uzz = hvp_fwdfwd(lambda z_: model(((xc, yc, z_), fc)), (zc,), (ones(zc),), True)
        lap = uxx + uyy + uzz
        ky = jnp.where(yc.reshape(-1) < yi, k1, k2)[None, None, :, None]        # k(y) sulle slice top/bottom
        f = fc.reshape(-1, nx, ny, 1)
        pde = jnp.mean(lap[:, 1:-1, 1:-1, 1:-1, :] ** 2)
        top = jnp.mean((ky * uz[:, :, :, -1, :] - f) ** 2)
        bottom = jnp.mean((u[:, :, :, 0, :] - a - b * ky * uz[:, :, :, 0, :]) ** 2)
        side = (jnp.mean(ux[:, 0] ** 2) + jnp.mean(ux[:, -1] ** 2)
                + jnp.mean(uy[:, :, 0] ** 2) + jnp.mean(uy[:, :, -1] ** 2))
        # flusso continuo attraverso l'interfaccia: derivate laterali a yi -/+ eps
        y_if = jnp.array([[yi - eps], [yi + eps]])
        uy_if = jax.jvp(lambda y_: model(((xc, y_, zc), fc)), (y_if,), (jnp.ones((2, 1)),))[1]
        ic = jnp.mean((k1 * uy_if[:, :, 0] - k2 * uy_if[:, :, 1]) ** 2)
        energy = jnp.mean((jnp.mean(ky[..., 0] * uz[:, :, :, -1, 0], axis=(1, 2)) - jnp.mean(ky[..., 0] * uz[:, :, :, 0, 0], axis=(1, 2))) ** 2)
        total = (lam['pde'] * pde + lam['top'] * top + lam['bottom'] * bottom + lam['side'] * side + lam['ic'] * ic
                 + lam['energy'] * energy)
        return total, dict(pde=pde, top=top, bottom=bottom, side=side, ic=ic, energy=energy)

    @eqx.filter_jit
    def loss_and_grad(model, xc, yc, zc, fc):
        (loss, aux), grads = eqx.filter_value_and_grad(terms, has_aux=True)(model, xc, yc, zc, fc)
        return loss, aux, grads
    return loss_and_grad


def pin_level(u, fc, p):
    """Fissa il livello: per Laplace con lati adiabatici la T media del fondo e' esattamente a + b*<f> (bilancio energetico).
    Aggiunge a ogni mappa la costante che impone questa media; una costante non altera PDE ne' flussi."""
    if not p.get('pin_level', False):
        return u
    target = p['bottom_a'] + p['bottom_b'] * jnp.mean(fc.reshape(fc.shape[0], -1), axis=1)      # (B,)
    shift = target - jnp.mean(u[:, :, :, 0], axis=(1, 2))
    return u + shift[:, None, None, None]


def make_loss_fd(p, nx, nz):
    """Loss fisica discretizzata (stile DeepOHeat-v2): residuo di div(k grad u) = 0 con stencil FD conservativo
    su una griglia densa nx*nx*nz, k sulle facce in y = media armonica (gestisce l'interfaccia senza termini ad hoc).
    Niente derivate autodiff: il modello non puo' 'barare' tra i punti perche' la fisica e' definita solo sui punti."""
    k1, k2, yi, a, b = p['k1'], p['k2'], p['y_interface'], p['bottom_a'], p['bottom_b']
    lam = p['lam']
    xg = jnp.linspace(0, 1, nx).reshape(-1, 1); zg = jnp.linspace(0, 0.5, nz).reshape(-1, 1)
    hx, hz = 1.0 / (nx - 1), 0.5 / (nz - 1)
    kc = jnp.where(xg.reshape(-1) < yi, k1, k2)                                   # k per cella lungo y (nx,)
    kf = 2 * kc[:-1] * kc[1:] / (kc[:-1] + kc[1:])                                # k sulle facce y (nx-1,)

    def terms(model, fc):
        f = jax.image.resize(fc.reshape(-1, 21, 21), (fc.shape[0], nx, nx), 'linear')[..., None]   # (B,nx,ny,1)
        u = pin_level(model(((xg, xg, zg), fc))[..., 0], fc, p)                   # (B,nx,ny,nz)
        # flussi conservativi: k * du sulle facce
        Fx = kc[None, None, :, None] * (u[:, 1:] - u[:, :-1]) / hx
        Fy = kf[None, None, :, None] * (u[:, :, 1:] - u[:, :, :-1]) / hx
        Fz = kc[None, None, :, None] * (u[:, :, :, 1:] - u[:, :, :, :-1]) / hz
        div = ((Fx[:, 1:] - Fx[:, :-1]) / hx)[:, :, 1:-1, 1:-1] \
            + ((Fy[:, :, 1:] - Fy[:, :, :-1]) / hx)[:, 1:-1, :, 1:-1] \
            + ((Fz[:, :, :, 1:] - Fz[:, :, :, :-1]) / hz)[:, 1:-1, 1:-1, :]
        pde = jnp.mean(div ** 2)
        uz_top = (3 * u[..., -1] - 4 * u[..., -2] + u[..., -3]) / (2 * hz)      # one-sided 2o ordine
        uz_bot = (-3 * u[..., 0] + 4 * u[..., 1] - u[..., 2]) / (2 * hz)
        top = jnp.mean((kc[None, None, :] * uz_top - f[..., 0]) ** 2)
        bottom = jnp.mean((u[..., 0] - a - b * kc[None, None, :] * uz_bot) ** 2)
        d1 = lambda v0, v1, v2, h: (-3 * v0 + 4 * v1 - v2) / (2 * h)
        side = (jnp.mean(d1(u[:, 0], u[:, 1], u[:, 2], hx) ** 2) + jnp.mean(d1(u[:, -1], u[:, -2], u[:, -3], hx) ** 2)
                + jnp.mean(d1(u[:, :, 0], u[:, :, 1], u[:, :, 2], hx) ** 2) + jnp.mean(d1(u[:, :, -1], u[:, :, -2], u[:, :, -3], hx) ** 2))
        # bilancio energetico per mappa: flusso medio top = flusso medio fondo
        energy = jnp.mean((jnp.mean(kc[None, None, :] * uz_top, axis=(1, 2)) - jnp.mean(kc[None, None, :] * uz_bot, axis=(1, 2))) ** 2)
        ic = jnp.zeros(())                                                        # gestita dalla media armonica
        total = lam['pde'] * pde + lam['top'] * top + lam['bottom'] * bottom + lam['side'] * side + lam['energy'] * energy
        return total, dict(pde=pde, top=top, bottom=bottom, side=side, ic=ic, energy=energy)

    @eqx.filter_jit
    def loss_and_grad(model, xc, yc, zc, fc):                                     # xc,yc,zc ignorati: griglia fissa
        (loss, aux), grads = eqx.filter_value_and_grad(terms, has_aux=True)(model, fc)
        return loss, aux, grads
    return loss_and_grad

@eqx.filter_jit
def update(grads, optimizer, opt_state, model):
    updates, opt_state = optimizer.update(grads, opt_state, model)
    return eqx.apply_updates(model, updates), opt_state


def evaluate(model, fs_test, u_test, result_dir, p, tag):
    x, y, z = (jnp.linspace(0, 1, 101).reshape(-1, 1), jnp.linspace(0, 1, 101).reshape(-1, 1),
               jnp.linspace(0, 0.5, 51).reshape(-1, 1))
    pred = np.asarray(pin_level(model(((x, y, z), fs_test))[..., 0], fs_test, p))
    true = np.asarray(u_test)
    np.save(os.path.join(result_dir, 'u_pred.npy'), pred)
    rel = [np.linalg.norm(pred[i] - true[i]) / np.linalg.norm(true[i]) for i in range(len(true))]
    peak = [25 * (pred[i].max() - true[i].max()) for i in range(len(true))]
    maxerr = [25 * np.abs(pred[i] - true[i]).max() for i in range(len(true))]
    # rilevatore di strato limite finto: curvatura in z vicino al top vs interno (griglia fine dz=0.01)
    dz = 0.5 / 50
    uzz = np.gradient(np.gradient(pred[0], dz, axis=2), dz, axis=2)
    bl_ratio = float(np.abs(uzz[:, :, 46:]).mean() / (np.abs(uzz[:, :, 5:46]).mean() + 1e-12))
    with open(os.path.join(result_dir, 'eval.txt'), 'w') as fh:
        fh.write(f'boundary-layer check |uzz| top(0.45-0.5)/interno: {bl_ratio:.1f}  (>10 = il modello bara tra i punti di collocazione)\n')
        for i in range(len(true)):
            fh.write(f'case {i}: rel_l2 {rel[i]:.4%}  peak_err {peak[i]:+.3f} C  max_abs_err {maxerr[i]:.3f} C\n')
        fh.write(f'mean rel_l2 {np.mean(rel):.4%}  mean |peak_err| {np.mean(np.abs(peak)):.3f} C\n')
    print(open(os.path.join(result_dir, 'eval.txt')).read())
    # figura sul primo caso
    T, P = 25 * true[0] + 20, 25 * pred[0] + 20
    kw = dict(extent=[0, 1, 0, 1], origin='lower')
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.4), constrained_layout=True)
    vmin, vmax = T.min(), T.max()
    im = ax[0].imshow(T[:, :, -1].T, cmap='magma', vmin=vmin, vmax=vmax, **kw); ax[0].set_title('T vera, top [°C]'); fig.colorbar(im, ax=ax[0])
    im = ax[1].imshow(P[:, :, -1].T, cmap='magma', vmin=vmin, vmax=vmax, **kw); ax[1].set_title(f'T predetta, top  (rel L2 {rel[0]:.2%})'); fig.colorbar(im, ax=ax[1])
    e = (P - T)[:, :, -1]; m = np.abs(e).max()
    im = ax[2].imshow(e.T, cmap='RdBu_r', vmin=-m, vmax=m, **kw); ax[2].set_title('errore [°C]'); fig.colorbar(im, ax=ax[2])
    ix = int(np.argmax(T[:, :, -1].max(axis=1)))
    yy = np.linspace(0, 1, 101)
    ax[3].plot(yy, T[ix, :, -1], 'k', lw=2, label='vera'); ax[3].plot(yy, P[ix, :, -1], 'r--', lw=2, label='predetta')
    ax[3].axvline(p['y_interface'], color='gray', ls=':', label=f"interfaccia y={p['y_interface']}")
    ax[3].set_title(f'profilo lungo y in superficie, x={ix/100:.2f}'); ax[3].set_xlabel('y'); ax[3].set_ylabel('T [°C]'); ax[3].legend()
    fig.suptitle(tag); fig.savefig(os.path.join(result_dir, 'eval_case0.png'), dpi=120); plt.close(fig)
    return rel, peak


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='kink', choices=['kink', 'v1'], help="v1 = stesso modello senza la feature |y-yi|")
    ap.add_argument('--k1', type=float, default=1.4); ap.add_argument('--k2', type=float, default=0.5)
    ap.add_argument('--y_interface', type=float, default=0.48)
    ap.add_argument('--bottom_a', type=float, default=0.2); ap.add_argument('--bottom_b', type=float, default=2.0)
    ap.add_argument('--eps_if', type=float, default=0.005)
    for t in ['pde', 'top', 'bottom', 'side', 'ic']:
        ap.add_argument(f'--lam_{t}', type=float, default=1.0)
    ap.add_argument('--lam_energy', type=float, default=0.0, help='peso del bilancio energetico top/fondo per mappa')
    ap.add_argument('--loss', default='pinn', choices=['pinn', 'fd'], help='pinn = residuo autodiff; fd = stencil discretizzato (stile DeepOHeat-v2)')
    ap.add_argument('--fd_nx', type=int, default=51); ap.add_argument('--fd_nz', type=int, default=26)
    ap.add_argument('--pin_level', type=int, default=0, help='1 = fissa analiticamente la T media del fondo a a+b*<f> (bilancio energetico)')
    ap.add_argument('--amp_aug_min', type=float, default=0.0, help='>0: meta\' delle mappe del batch scalate per un fattore log-uniforme in [amp_aug_min, 1]')
    ap.add_argument('--train_data', default='data/fs_train_surface_mixed_p4.npy')
    ap.add_argument('--test_f', default='data/fs_test_11.npy', help='uno o piu\' file separati da virgola')
    ap.add_argument('--test_u', default='data/u_test_11.npy', help='stesso ordine di --test_f')
    ap.add_argument('--epochs', type=int, default=50000); ap.add_argument('--decay_steps', type=int, default=1000)
    ap.add_argument('--lr', type=float, default=1e-3); ap.add_argument('--batch', type=int, default=50)
    ap.add_argument('--nc', type=int, default=21); ap.add_argument('--log_epoch', type=int, default=500)
    ap.add_argument('--nz', type=int, default=21, help='punti di collocazione in z (estremi inclusi)')
    ap.add_argument('--z_random', type=int, default=1, help='1 = punti interni in z campionati a caso a ogni passo (evita strati limite finti tra i punti)')
    ap.add_argument('--seed', type=int, default=42); ap.add_argument('--tag', default='')
    ap.add_argument('--branch_depth', type=int, default=8); ap.add_argument('--branch_hidden', type=int, default=256)
    ap.add_argument('--trunk_depth', type=int, default=3); ap.add_argument('--trunk_hidden', type=int, default=64)
    ap.add_argument('--r', type=int, default=128)
    args = ap.parse_args()
    assert args.nc == 21, 'la mappa f e\' definita sui 21x21 punti sensore: nc deve essere 21'

    p = dict(k1=args.k1, k2=args.k2, y_interface=args.y_interface, bottom_a=args.bottom_a, bottom_b=args.bottom_b, pin_level=bool(args.pin_level),
             eps_if=args.eps_if, lam=dict(pde=args.lam_pde, top=args.lam_top, bottom=args.lam_bottom,
                                          side=args.lam_side, ic=args.lam_ic, energy=args.lam_energy))
    fs_train = jnp.asarray(np.load(args.train_data).reshape(-1, 441).astype(np.float32))
    tests = [(jnp.asarray(np.load(f).reshape(-1, 441).astype(np.float32)), np.load(u).astype(np.float32).reshape(-1, 101, 101, 51), os.path.basename(f))
             for f, u in zip(args.test_f.split(','), args.test_u.split(','))]

    tag = f'{args.model}{args.tag}'
    result_dir = os.path.join('results', 'results_2mat', tag); os.makedirs(result_dir, exist_ok=True)
    with open(os.path.join(result_dir, 'args.txt'), 'w') as fh:
        fh.write('\n'.join(f'{k}={v}' for k, v in vars(args).items()))

    key = jax.random.PRNGKey(args.seed); key, sub = jax.random.split(key)
    model = DeepOHeat_kink(441, y_interface=args.y_interface, use_kink=(args.model == 'kink'),
                           branch_depth=args.branch_depth, branch_hidden=args.branch_hidden,
                           trunk_depth=args.trunk_depth, trunk_hidden=args.trunk_hidden, rank=args.r, key=sub)
    optimizer = optax.adam(optax.exponential_decay(args.lr, args.decay_steps, 0.9))
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    loss_and_grad = make_loss(p) if args.loss == 'pinn' else make_loss_fd(p, args.fd_nx, args.fd_nz)

    xc = jnp.linspace(0, 1, args.nc).reshape(-1, 1); yc = xc
    z_fixed = jnp.linspace(0, 0.5, args.nz).reshape(-1, 1)
    def sample_z(k):
        if not args.z_random:
            return z_fixed
        zi = jax.random.uniform(k, (args.nz - 2,), minval=0.0, maxval=0.5)
        return jnp.sort(jnp.concatenate([jnp.array([0.0, 0.5]), zi])).reshape(-1, 1)
    log = open(os.path.join(result_dir, 'loss_terms.csv'), 'w'); log.write('epoch,total,pde,top,bottom,side,ic,energy\n')
    t0 = time.time()
    for epoch in range(args.epochs):
        key, sub, subz = jax.random.split(key, 3)
        fc = fs_train[jax.random.choice(sub, fs_train.shape[0], (args.batch,), replace=False)]
        if args.amp_aug_min > 0:
            key, k1_, k2_ = jax.random.split(key, 3)
            scale = jnp.exp(jax.random.uniform(k1_, (args.batch, 1), minval=jnp.log(args.amp_aug_min), maxval=0.0))
            scale = jnp.where(jax.random.bernoulli(k2_, 0.5, (args.batch, 1)), scale, 1.0)
            fc = fc * scale
        zc = sample_z(subz)
        loss, aux, grads = loss_and_grad(model, xc, yc, zc, fc)
        model, opt_state = update(grads, optimizer, opt_state, model)
        if epoch % args.log_epoch == 0 or epoch == args.epochs - 1:
            a = {k: float(v) for k, v in aux.items()}
            print(f"Epoch {epoch+1}/{args.epochs} total {float(loss):.3e} | pde {a['pde']:.2e} top {a['top']:.2e} "
                  f"bottom {a['bottom']:.2e} side {a['side']:.2e} ic {a['ic']:.2e} energy {a['energy']:.2e}", flush=True)
            log.write(f"{epoch+1},{float(loss)},{a['pde']},{a['top']},{a['bottom']},{a['side']},{a['ic']},{a['energy']}\n"); log.flush()
    log.close()
    runtime = time.time() - t0
    print(f'Runtime {runtime:.1f}s ({1000*runtime/args.epochs:.2f} ms/iter)')
    eqx.tree_serialise_leaves(os.path.join(result_dir, 'model.eqx'), model)
    for i, (fs_t, u_t, name) in enumerate(tests):
        d = result_dir if i == 0 else os.path.join(result_dir, f'eval_{name.replace(".npy", "")}')
        os.makedirs(d, exist_ok=True)
        print(f'--- test {name} ---')
        evaluate(model, fs_t, u_t, d, p, f'{tag} — {name}')
