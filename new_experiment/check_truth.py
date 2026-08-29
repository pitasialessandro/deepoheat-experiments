"""Verifica con differenze finite che la ground truth Ansys soddisfi la loss a due materiali.

Da lanciare PRIMA di allenare: se i residui non sono ~0, la loss descrive un problema diverso da quello
risolto da Ansys e nessun training puo' convergere alla soluzione vera.
"""
import argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--u', default='data/u_test_11.npy', help='temperatura normalizzata (1,101,101,51)')
ap.add_argument('--f', default='data/fs_test_11.npy', help='mappa di potenza (1,441)')
ap.add_argument('--k1', type=float, default=1.4)
ap.add_argument('--k2', type=float, default=0.5)
ap.add_argument('--y_interface', type=float, default=0.48)
ap.add_argument('--bottom_a', type=float, default=0.2, help='u_fondo - a = b * k * uz_fondo')
ap.add_argument('--bottom_b', type=float, default=2.0)
args = ap.parse_args()

u = np.load(args.u).reshape(101, 101, 51).astype(np.float64)
f21 = np.load(args.f).reshape(21, 21)
nx, ny, nz = u.shape
x, y, z = np.linspace(0, 1, nx), np.linspace(0, 1, ny), np.linspace(0, 0.5, nz)
dx, dy, dz = x[1] - x[0], y[1] - y[0], z[1] - z[0]
K = np.where(y < args.y_interface, args.k1, args.k2)           # (ny,)

ux, uy, uz = np.gradient(u, dx, axis=0), np.gradient(u, dy, axis=1), np.gradient(u, dz, axis=2)
lap = np.gradient(ux, dx, axis=0) + np.gradient(uy, dy, axis=1) + np.gradient(uz, dz, axis=2)
uz_top = (3 * u[:, :, -1] - 4 * u[:, :, -2] + u[:, :, -3]) / (2 * dz)
uz_bot = (-3 * u[:, :, 0] + 4 * u[:, :, 1] - u[:, :, 2]) / (2 * dz)
rms = lambda a: float(np.sqrt(np.mean(a ** 2)))

# mappa f sulla griglia fine: ogni cella sensore copre 5x5 punti
f101 = f21[np.clip(np.round(np.arange(nx) / 5).astype(int), 0, 20)][:, np.clip(np.round(np.arange(ny) / 5).astype(int), 0, 20)]

j = int(np.argmin(np.abs(y - args.y_interface)))
band = np.abs(y - args.y_interface) > 0.03                      # escludo il kink dal laplaciano
inner = (slice(3, -3), band, slice(3, -3))
r1 = y[band] < args.y_interface
lap_in = lap[3:-3][:, band][:, :, 3:-3]
uy_m, uy_p = (u[:, j - 1, :] - u[:, j - 3, :]) / (2 * dy), (u[:, j + 3, :] - u[:, j + 1, :]) / (2 * dy)

print(f'file: {args.u}   scala di riferimento: RMS(uzz) = {rms(np.gradient(uz, dz, axis=2)):.3f}, max f = {f21.max():.3f}')
print(f'PDE  laplaciano  regione1 RMS {rms(lap_in[:, r1]):.4f}   regione2 RMS {rms(lap_in[:, ~r1]):.4f}')
print(f'TOP  k*uz - f    RMS {rms(K * uz_top - f101):.4f}   (dentro riscaldatore: k*uz medio {(K*uz_top)[f101>0].mean():.3f} vs f {f101[f101>0].mean():.3f})')
print(f'BOT  u - a - b*k*uz   RMS {rms(u[:, :, 0] - args.bottom_a - args.bottom_b * K * uz_bot):.4f}   (u_fondo medio {u[:,:,0].mean():.4f})')
print(f'SIDE ux(0), ux(1), uy(0), uy(1)  RMS {rms(ux[0]):.4f} {rms(ux[-1]):.4f} {rms(uy[:,0]):.4f} {rms(uy[:,-1]):.4f}')
print(f'INTERFACCIA y={y[j]:.2f}: salto di k*uy  RMS {rms(args.k1 * uy_m - args.k2 * uy_p):.4f}   (RMS di k*uy stesso {rms(args.k1*uy_m):.4f}); '
      f'con k uniforme sarebbe RMS {rms(uy_m - uy_p):.4f}')
