import numpy as np

# ============================================================
# Config
# ============================================================
FLUX_FILE = "data/flusso_matrice_14.npy"
TEMP_FILE = "data/temperatura_matrice_11.npy"

SAVE_FS_PAPER = "data/fs_test_paper_like.npy"   # 100000 -> 40
SAVE_U_TEST   = "data/u_test_custom.npy"

# ============================================================
# 1. Carica flux e prendi la faccia superiore
# ============================================================
flux = np.load(FLUX_FILE)
print("flux shape originale:", flux.shape)

# assumo shape (21, 21, 11) con z ultimo asse
flux_top = flux
print("flux top shape:", flux_top.shape)

# opzionale: pulizia di nan
flux_top = np.nan_to_num(flux_top, nan=0.0)

print("Valori unici raw top flux:", np.unique(flux_top))

# ------------------------------------------------------------
# B) PAPER-LIKE:
# conversione fisica rigorosa usando:
# tile area = (1 mm / 20)^2 = (0.05e-3)^2 = 2.5e-9 m^2
# power scale per tile = 0.00625 mW = 6.25e-6 W
#
# level = q * A_tile / P0
#       = q * 2.5e-9 / 6.25e-6
#       = q * 4e-4
#
# quindi 100000 -> 40
# ------------------------------------------------------------
A_tile = 2.5e-9       # m^2
P0 = 6.25e-6          # W
#fs_paper_like = flux_top * A_tile / P0
fs_paper_like = np.zeros((21,21),dtype=np.float32)
#fs_paper_like[0:10,0:10] = 1.0
#fs_paper_like[:,:] = 0.5
#fs_paper_like[16:24,10:32] = 1.0
fs_paper_like[8:12,5:16] = 1.0

print("Valori unici paper-like:", np.unique(fs_paper_like))

# reshape come richiesto dal codice: (N, 21*21)
fs_paper_like = fs_paper_like.reshape(1, 21 * 21).astype(np.float32)

# salva entrambe
np.save(SAVE_FS_PAPER, fs_paper_like)

print("Salvato:", SAVE_FS_PAPER, fs_paper_like.shape)

# ============================================================
# 3. Carica temperatura e riscala
# ============================================================
Temp = np.load(TEMP_FILE)
print("Temp shape originale:", Temp.shape)

# opzionale: pulizia di nan
Temp = np.nan_to_num(Temp, nan=0.0)

print("Temp range in °C:", Temp.min(), Temp.max())

# Il modello usa:
# T_phys = 25*u + 293.15
# Se Temperatura è in °C:
# T_K = T_C + 273.15
# u = (T_K - 293.15)/25 = (T_C - 20)/25
u_test = (Temp - 20.0) / 25.0

print("u_test range scalato:", u_test.min(), u_test.max())

# reshape come richiesto dall'eval:
# (N, 101, 101, 51) oppure spesso anche (N, 101, 101, 51, 1)
# dal tuo eval sembra basti che il batch axis ci sia
u_test = u_test.reshape(1, 101, 101, 51).astype(np.float32)

np.save(SAVE_U_TEST, u_test)
print("Salvato:", SAVE_U_TEST, u_test.shape)