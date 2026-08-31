"""Genera i test set .npz per heat_surface4.py (chiavi: f, u, k1, k2, yi).

- ktest_ansys11/14.npz: i due casi Ansys (input = mappe a flusso esatto), k = quello vero.
- ktest_var.npz: casi risolti con fd_solver su k mai visti (6 configurazioni x 2 mappe held-out
  di fs_test_surface.npy, scalate a 0.25 per stare su ampiezze fisiche simili ad Ansys).

Eseguire dalla root: python new_experiment/make_k_testsets.py
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from fd_solver import solve_k_step

A, B = 0.2, 2.0

for case in ('11', '14'):
    f = np.load(f'data/fs_test_{case}_flux.npy').reshape(1, 441).astype(np.float32)
    u = np.load(f'data/u_test_{case}.npy').reshape(1, 101, 101, 51).astype(np.float32)
    np.savez(f'data/ktest_ansys{case}.npz', f=f, u=u,
             k1=np.array([1.4], np.float32), k2=np.array([0.5], np.float32), yi=np.array([0.48], np.float32))
    print(f'ktest_ansys{case}.npz: 1 caso')

configs = [(1.4, 0.5, 0.48), (1.4, 0.5, 0.30), (0.5, 1.4, 0.60),
           (2.0, 0.8, 0.35), (0.7, 0.35, 0.72), (1.0, 1.0, 0.50)]
maps = np.load('data/fs_test_surface.npy').astype(np.float64)[[0, 7]] * 0.25

F, U, K1, K2, YI = [], [], [], [], []
t0 = time.time()
for k1, k2, yi in configs:
    for f21 in maps:
        u, res = solve_k_step(f21, k1, k2, yi, a=A, b=B)
        assert res < 1e-8, res
        F.append(f21.reshape(441)); U.append(u); K1.append(k1); K2.append(k2); YI.append(yi)
        print(f'k1={k1} k2={k2} yi={yi}: u in [{u.min():.3f}, {u.max():.3f}]  ({time.time()-t0:.0f}s)')
np.savez('data/ktest_var.npz', f=np.array(F, np.float32), u=np.array(U, np.float32),
         k1=np.array(K1, np.float32), k2=np.array(K2, np.float32), yi=np.array(YI, np.float32))
print(f'ktest_var.npz: {len(F)} casi in {time.time()-t0:.0f}s')
