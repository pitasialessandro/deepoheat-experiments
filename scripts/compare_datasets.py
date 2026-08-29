import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = 'results/results_surface/DeepOHeat_v1/nf50_nc21_branch_8_256_trunk_3_64_r128'
RUNS = [('solo GRF', f'{BASE}_ep50000_d1000'), ('misto (pmax 2)', f'{BASE}_mixed'), ('misto (pmax 4)', f'{BASE}_mixed_p4'), ('100% blocchi (pmax 4)', f'{BASE}_blocks_p4')]
u = np.load('data/u_test_surface.npy'); fs = np.load('data/fs_test_surface.npy')
preds = [np.load(f'{d}/u_pred_heat3d.npy')[..., 0] for _, d in RUNS]
rl2 = lambda a, b: np.linalg.norm(a - b) / np.linalg.norm(a)
print('mappa                 ', '  '.join(f'{i:>6}' for i in range(10)), '   media')
for (name, _), p in zip(RUNS, preds):
    r = [rl2(u[i], p[i]) for i in range(10)]
    print(f'{name:<22}', '  '.join(f'{x:6.2%}' for x in r), f'  {np.mean(r):6.2%}')
    pk = [25*(p[i].max()-u[i].max()) for i in range(10)]
    print(f'{"  err. picco [°C]":<22}', '  '.join(f'{x:+6.2f}' for x in pk), f'  {np.mean(np.abs(pk)):6.2f} (|.| medio)')

samples = [0, 9, 5]
fig, ax = plt.subplots(len(samples), 2 + len(RUNS), figsize=(4.2*(2 + len(RUNS)), 3.8*len(samples)), constrained_layout=True)
kw = dict(extent=[0,1,0,1], origin='lower')
for r_, s in enumerate(samples):
    T = 25*u[s]+20; Ps = [25*p[s]+20 for p in preds]
    vmin, vmax = T.min(), T.max()
    im = ax[r_,0].imshow(fs[s].T, cmap='Greys', **kw); ax[r_,0].set_title(f'test[{s}] potenza (max {fs[s].max():g})'); fig.colorbar(im, ax=ax[r_,0], shrink=0.8)
    im = ax[r_,1].imshow(T[:,:,-1].T, cmap='magma', vmin=vmin, vmax=vmax, **kw); ax[r_,1].set_title('T vera, top [°C]'); fig.colorbar(im, ax=ax[r_,1], shrink=0.8)
    for k, ((name, _), P) in enumerate(zip(RUNS, Ps)):
        im = ax[r_,2+k].imshow(P[:,:,-1].T, cmap='magma', vmin=vmin, vmax=vmax, **kw)
        ax[r_,2+k].set_title(f'{name}  (L2 {rl2(u[s], preds[k][s]):.2%})'); fig.colorbar(im, ax=ax[r_,2+k], shrink=0.8)
fig.suptitle('Superficie top z = 0.5 — stesso modello e iperparametri, cambia solo il training set')
out = f'{BASE}_mixed_p4/datasets_comparison.png'; fig.savefig(out, dpi=110); print(out)
