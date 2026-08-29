# Risultati

Ogni cartella contiene metriche (`eval*.txt` o `log (eval metrics).csv`), log della loss, figure e i pesi del
modello (`*.eqx`). Le predizioni `u_pred*.npy` (20 MB ciascuna) non sono incluse: si rigenerano dai pesi.

## `surface/` — caso del paper, modello `DeepOHeat_v1`, 10 mappe di test

| cartella | setup | MAPE | rel L2 |
|---|---|---|---|
| `01_default_10k` | default repo (= paper: 10k ep, decay 500) | 0.058% | 2.96% |
| `02_50k_decay500` | 50k epoche | — | 2.04% |
| `03_50k_decay1000` | 50k epoche, decay 1000 | 0.043% | 1.96% |
| `04_mixed_pmax2` | + training 50% GRF + 50% blocchi (pmax 2) | 0.028% | 1.14% |
| `05_blocks_pmax4` | training 100% blocchi (pmax 4) — fallisce | 0.110% | 4.42% |
| `06_mixed_pmax4` | training 50/50, pmax 4 — **migliore** | 0.031% | 1.10% |

`edge_profile.png`: la soluzione vera è smussata al bordo di un blocco. `train_vs_test_maps.png`: GRF vs blocchi.

## `two_materials/` — chip a due materiali, casi Ansys 11 e 14

| cartella | loss / modello | caso 11 | note |
|---|---|---|---|
| `A_stepA_paper_sanity` | FD pipeline con k1 = k2 = 1 sul caso del paper | 3.5% (10 mappe) | pipeline ok |
| `B0_pinn_zfixed_CHEAT` | PINN, z fissa | 33% | strato limite finto al top |
| `B1_pinn_zrandom` | PINN, z casuale | 30% | forma giusta, livello −2 °C |
| `C_pinn_v1_nokink` | PINN, senza feature `\|y−y_i\|` | 24% | interfaccia 300× peggio |
| `D0_fd` | **FD discretizzata**, 30k ep | 1.34% (0.47% con input da dati) | primo run funzionante |
| `D1_fd_aug` | FD + augmentation ampiezza, 50k ep | 3.0% | livello variabile |
| `D3_fd_aug_pin` | FD + augmentation + `--pin_level` | **0.72%** (input `_flux`), caso 14 **1.33%** | max errore 0.22 / 0.47 °C |

Le sottocartelle `eval_*` contengono la valutazione dello stesso modello su input diversi (`_data`, `_cons`, `_flux`, caso 14).
`diagnostica/`: verifica della ground truth Ansys (posizione del riscaldatore, kink all'interfaccia, BC di fondo).
