"""Genera mappe di potenza 21x21 a blocchi rettangolari e un training set misto GRF + blocchi.

Le mappe di test del paper sono layout a rettangoli con bordi netti, mentre fs_train_surface.npy
contiene solo campi gaussiani lisci: questo script produce un training set che copre entrambi.
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

N = 21           # punti sensore per asse (griglia 20x20 tile, punti in i/20)
SS = 5           # supersampling per cella: da' valori frazionari sui bordi, come nei test


def block_map(rng, pmax):
    # coverage frazionaria: ogni punto sensore rappresenta la cella [i/20 - 1/40, i/20 + 1/40]
    fine = np.zeros((N * SS, N * SS))
    edges = (np.arange(N * SS) + 0.5) / SS / (N - 1) - 0.5 / (N - 1)   # coordinate dei sub-pixel
    X, Y = np.meshgrid(edges, edges, indexing='ij')

    if rng.random() < 0.05:                       # potenza uniforme (come fs_test[3])
        return np.full((N, N), rng.uniform(0.25, pmax))

    for _ in range(rng.integers(1, 7)):
        w, h = rng.uniform(0.1, 0.5, size=2)
        x0, y0 = rng.uniform(0, 1 - w), rng.uniform(0, 1 - h)
        p = np.exp(rng.uniform(np.log(0.25), np.log(pmax)))
        inside = (X >= x0) & (X <= x0 + w) & (Y >= y0) & (Y <= y0 + h)
        fine = np.where(inside, np.maximum(fine, p), fine)   # sovrapposizioni: vince la piu' alta
    return fine.reshape(N, SS, N, SS).mean(axis=(1, 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_blocks', type=int, default=5000)
    ap.add_argument('--n_grf', type=int, default=5000, help='campi gaussiani presi da fs_train_surface.npy')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--pmax', type=float, default=2.0, help='potenza massima dei blocchi')
    ap.add_argument('--out', default='data/fs_train_surface_mixed.npy')
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    blocks = np.stack([block_map(rng, args.pmax) for _ in range(args.n_blocks)])
    grf_all = np.load('data/fs_train_surface.npy')
    grf = grf_all[rng.choice(len(grf_all), args.n_grf, replace=False)]
    mixed = np.concatenate([grf, blocks])
    rng.shuffle(mixed)
    np.save(args.out, mixed)
    print(f'{args.out}: {mixed.shape}, blocchi range [{blocks.min():.2f}, {blocks.max():.2f}], '
          f'media blocchi {blocks.mean():.3f}, n_grf {len(grf)}')

    fig, ax = plt.subplots(2, 6, figsize=(16, 5.6), constrained_layout=True)
    for k in range(6):
        ax[0, k].imshow(blocks[k].T, origin='lower', cmap='Greys', vmin=0, vmax=args.pmax); ax[0, k].set_title(f'blocco #{k}')
        ax[1, k].imshow(blocks[6 + k].T if len(grf) == 0 else grf[k].T, origin='lower', cmap='Greys', vmin=0 if len(grf) == 0 else -3, vmax=args.pmax if len(grf) == 0 else 3)
        ax[1, k].set_title(f'blocco #{6 + k}' if len(grf) == 0 else f'GRF #{k}')
    fig.suptitle('Esempi dal training set misto (sopra: blocchi generati, sotto: campi gaussiani originali)')
    prev = args.out.replace('.npy', '_preview.png'); fig.savefig(prev, dpi=110); print(prev)


if __name__ == '__main__':
    main()
