"""Solutore diretto di div(k grad u) = 0 sul dominio 1x1x0.5, stesse BC della loss:
top k*uz = f, fondo u - a = b*k*uz (Robin), lati adiabatici. k = k(y) per nodo, media armonica sulle facce.

Serve a generare test set illimitati con k variabile (il modello con k in input non puo' essere
validato solo sui due casi Ansys, che hanno un unico k). Discretizzazione ai nodi 101x101x51,
pesi di volume ai bordi -> matrice simmetrica, risolta con CG precondizionato AMG (pyamg).

Uso:
  python new_experiment/fd_solver.py --validate            # confronto con Ansys, casi 11 e 14
  (come modulo) from fd_solver import solve_k_step, f21_to_grid
"""
import os, sys, argparse, time
import numpy as np
import scipy.sparse as sp
import pyamg


def k_step_profile(y, k1, k2, yi):
    return np.where(y < yi, k1, k2)


def f21_to_grid(f21, nx, ny):
    """Mappa 21x21 -> griglia fine, costante a tratti per cella sensore (come check_truth.py)."""
    ix = np.clip(np.round(np.arange(nx) / ((nx - 1) / 20)).astype(int), 0, 20)
    iy = np.clip(np.round(np.arange(ny) / ((ny - 1) / 20)).astype(int), 0, 20)
    return f21[ix][:, iy]


def solve(f_grid, k_nodes, a=0.2, b=2.0, nx=101, ny=101, nz=51, lz=0.5, tol=1e-10, k_faces=None):
    """f_grid: (nx,ny) flusso al top nelle unita' del codice; k_nodes: (ny,) k per nodo lungo y
    (usato per i flussi x/z e le BC); k_faces: (ny-1,) k sulle facce y (default: media armonica dei nodi).
    Ritorna u (nx,ny,nz) normalizzata (T = 25*u + T_amb)."""
    dx, dy, dz = 1.0 / (nx - 1), 1.0 / (ny - 1), lz / (nz - 1)
    N = nx * ny * nz
    idx = lambda i, j, l: (i * ny + j) * nz + l

    ky = np.asarray(k_nodes, dtype=np.float64)                    # (ny,)
    kf = 2 * ky[:-1] * ky[1:] / (ky[:-1] + ky[1:]) if k_faces is None else np.asarray(k_faces, np.float64)

    # pesi di volume (0.5 su ogni bordo toccato) -> matrice simmetrica
    wx = np.ones(nx); wx[[0, -1]] = 0.5
    wy = np.ones(ny); wy[[0, -1]] = 0.5
    wz = np.ones(nz); wz[[0, -1]] = 0.5
    W = wx[:, None, None] * wy[None, :, None] * wz[None, None, :]  # (nx,ny,nz)

    I, J, V = [], [], []
    rhs = np.zeros(N)

    def add(r, c, v):
        I.append(r); J.append(c); V.append(v)

    Ii, Jj, Ll = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing='ij')
    Ii, Jj, Ll = Ii.ravel(), Jj.ravel(), Ll.ravel()
    rows = idx(Ii, Jj, Ll)
    w = W.ravel()
    kc = ky[Jj]                                                    # k al nodo (dipende solo da j)

    diag = np.zeros(N)

    # --- direzione x (k costante lungo x): facce i-1/2 e i+1/2, mirror ai bordi ---
    # bordo mirror: il nodo di bordo vede il vicino con coeff 2k/h^2, il peso 0.5 lo riporta a k/h^2 (simmetria)
    for step in (-1, 1):
        m = (Ii + step >= 0) & (Ii + step <= nx - 1)
        cb = np.where((Ii[m] == 0) | (Ii[m] == nx - 1), 2.0, 1.0)
        val = w[m] * kc[m] * cb / dx ** 2
        I.extend(rows[m]); J.extend(idx(Ii[m] + step, Jj[m], Ll[m])); V.extend(val)
        np.add.at(diag, rows[m], -val)

    # --- direzione y (k armonico sulle facce), mirror ai bordi ---
    for step in (-1, 1):
        m = (Jj + step >= 0) & (Jj + step <= ny - 1)
        jface = np.where(step == -1, Jj[m] - 1, Jj[m])             # indice faccia tra j e j+step
        kface = kf[jface]
        cb = np.where((Jj[m] == 0) | (Jj[m] == ny - 1), 2.0, 1.0)
        val = w[m] * kface * cb / dy ** 2
        I.extend(rows[m]); J.extend(idx(Ii[m], Jj[m] + step, Ll[m])); V.extend(val)
        np.add.at(diag, rows[m], -val)

    # --- direzione z (k costante lungo z) ---
    for step in (-1, 1):
        m = (Ll + step >= 0) & (Ll + step <= nz - 1)
        cb = np.where((Ll[m] == 0) | (Ll[m] == nz - 1), 2.0, 1.0)
        val = w[m] * kc[m] * cb / dz ** 2
        I.extend(rows[m]); J.extend(idx(Ii[m], Jj[m], Ll[m] + step)); V.extend(val)
        np.add.at(diag, rows[m], -val)

    # --- BC top (l = nz-1): ghost u_g = u[nz-2] + 2 dz f / k -> termine noto 2 f / dz (pesato) ---
    mtop = Ll == nz - 1
    rhs[rows[mtop]] += -w[mtop] * 2.0 * f_grid[Ii[mtop], Jj[mtop]] / dz   # segno: sistema -div, vedi sotto

    # --- BC fondo (l = 0), Robin: u - a = b k uz -> ghost u_g = u[1] - 2 dz (u0 - a)/(b k) ---
    mbot = Ll == 0
    coef = w[mbot] * 2.0 / (b * dz)
    np.add.at(diag, rows[mbot], -coef)
    rhs[rows[mbot]] += -coef * a

    I.extend(rows); J.extend(rows); V.extend(diag)
    A = sp.csr_matrix((V, (I, J)), shape=(N, N))
    A = -A                                                          # def. positiva
    rhs = -rhs

    ml = pyamg.smoothed_aggregation_solver(A, max_coarse=500)
    x0 = np.full(N, a + b * float(f_grid.mean()))
    res = []
    u = ml.solve(rhs, x0=x0, tol=tol, accel='cg', residuals=res, maxiter=300)
    rel_res = float(np.linalg.norm(A @ u - rhs) / np.linalg.norm(rhs))
    return u.reshape(nx, ny, nz), rel_res


def solve_k_step(f21, k1, k2, yi, a=0.2, b=2.0, nx=101, ny=101, nz=51):
    """Interfaccia sub-griglia: facce y = media armonica pesata per la frazione theta,
    nodi = media di volume (come la loss di heat_surface4)."""
    y = np.linspace(0, 1, ny)
    h = 1.0 / (ny - 1)
    th_f = np.clip((yi - y[:-1]) / h, 0.0, 1.0)
    kf = 1.0 / (th_f / k1 + (1 - th_f) / k2)
    lo, hi = np.clip(y - h / 2, 0, 1), np.clip(y + h / 2, 0, 1)
    th_c = np.clip((yi - lo) / (hi - lo), 0.0, 1.0)
    kc = th_c * k1 + (1 - th_c) * k2
    f_grid = f21_to_grid(np.asarray(f21, dtype=np.float64).reshape(21, 21), nx, ny)
    return solve(f_grid, kc, a=a, b=b, nx=nx, ny=ny, nz=nz, k_faces=kf)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--validate', action='store_true', help='confronta con la ground truth Ansys (casi 11 e 14)')
    ap.add_argument('--k1', type=float, default=1.4); ap.add_argument('--k2', type=float, default=0.5)
    ap.add_argument('--y_interface', type=float, default=0.48)
    ap.add_argument('--bottom_a', type=float, default=0.2); ap.add_argument('--bottom_b', type=float, default=2.0)
    args = ap.parse_args()

    if args.validate:
        y = np.linspace(0, 1, 101)
        K = k_step_profile(y, args.k1, args.k2, args.y_interface)
        for case in ('11', '14'):
            u_true = np.load(f'data/u_test_{case}.npy').reshape(101, 101, 51).astype(np.float64)
            dz = 0.5 / 50
            uz_top = (3 * u_true[:, :, -1] - 4 * u_true[:, :, -2] + u_true[:, :, -3]) / (2 * dz)
            f_exact = K[None, :] * uz_top                       # BC vera ricavata dai dati
            t0 = time.time()
            u1, r1 = solve(f_exact, K, a=args.bottom_a, b=args.bottom_b)
            f21 = np.load(f'data/fs_test_{case}_flux.npy').reshape(21, 21)
            u2, r2 = solve_k_step(f21, args.k1, args.k2, args.y_interface, a=args.bottom_a, b=args.bottom_b)
            dt = time.time() - t0
            for name, u_s, rr in (('f esatta (k*uz dai dati)', u1, r1), ('f = mappa 21x21 flux', u2, r2)):
                rel = np.linalg.norm(u_s - u_true) / np.linalg.norm(u_true)
                mx = 25 * np.abs(u_s - u_true).max()
                off = 25 * (u_s - u_true).mean()
                print(f'caso {case} [{name}]  rel L2 {rel:.4%}  max {mx:.3f} C  offset medio {off:+.3f} C  (residuo {rr:.1e})')
            print(f'  tempo per i 2 solve: {dt:.1f}s')
