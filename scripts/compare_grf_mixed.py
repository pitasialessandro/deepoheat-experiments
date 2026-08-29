import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = 'results/results_surface/DeepOHeat_v1/nf50_nc21_branch_8_256_trunk_3_64_r128'
RUNS = [('solo GRF', f'{BASE}_ep50000_d1000'), ('misto GRF + blocchi', f'{BASE}_mixed')]
u = np.load('data/u_test_surface.npy'); fs = np.load('data/fs_test_surface.npy')
preds = [np.load(f'{d}/u_pred_heat3d.npy')[..., 0] for _, d in RUNS]
rl2 = lambda a, b: np.linalg.norm(a - b) / np.linalg.norm(a)
print('mappa      ', '  '.join(f'{i:>6}' for i in range(10)), '   media')
for (name, _), p in zip(RUNS, preds):
    r = [rl2(u[i], p[i]) for i in range(10)]
    print(f'{name:<20}', '  '.join(f'{x:6.2%}' for x in r), f'  {np.mean(r):6.2%}')
    pk = [25*(p[i].max()-u[i].max()) for i in range(10)]
    print(f'{"  err. picco [°C]":<20}', '  '.join(f'{x:+6.2f}' for x in pk), f'  {np.mean(np.abs(pk)):6.2f} (|.| medio)')

# figura: 3 mappe di test (0, 1, 6), colonne: potenza | vera | pred GRF | pred misto | err GRF | err misto
samples = [0, 1, 6]
fig, ax = plt.subplots(len(samples), 6, figsize=(22, 3.6*len(samples)), constrained_layout=True)
kw = dict(extent=[0,1,0,1], origin='lower')
for r_, s in enumerate(samples):
    T = 25*u[s]+20; Ps = [25*p[s]+20 for p in preds]
    vmin, vmax = T.min(), T.max(); emax = max(np.abs(P-T).max() for P in Ps)
    im = ax[r_,0].imshow(fs[s].T, cmap='Greys', **kw); ax[r_,0].set_title(f'test[{s}] potenza'); fig.colorbar(im, ax=ax[r_,0], shrink=0.8)
    im = ax[r_,1].imshow(T[:,:,-1].T, cmap='magma', vmin=vmin, vmax=vmax, **kw); ax[r_,1].set_title('T vera, top [°C]'); fig.colorbar(im, ax=ax[r_,1], shrink=0.8)
    for k, ((name, _), P) in enumerate(zip(RUNS, Ps)):
        im = ax[r_,2+k].imshow(P[:,:,-1].T, cmap='magma', vmin=vmin, vmax=vmax, **kw)
        ax[r_,2+k].set_title(f'pred {name}  (L2 {rl2(u[s], preds[k][s]):.2%})'); fig.colorbar(im, ax=ax[r_,2+k], shrink=0.8)
        im = ax[r_,4+k].imshow((P-T)[:,:,-1].T, cmap='RdBu_r', vmin=-emax, vmax=emax, **kw)
        ax[r_,4+k].set_title(f'errore {name}'); fig.colorbar(im, ax=ax[r_,4+k], shrink=0.8)
fig.suptitle('Superficie top z = 0.5 — stesso modello e iperparametri (50k ep., decay 1000), cambia solo il training set')
out = f'{BASE}_mixed/grf_vs_mixed.png'; fig.savefig(out, dpi=110); print(out)
