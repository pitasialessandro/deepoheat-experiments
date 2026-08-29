# DeepOHeat-v1 — esperimenti

Esperimenti sul codice di [DeepOHeat-v1](https://github.com/xlyu0127/DeepOHeat-v1) (Yu et al., IEEE TCPMT 2025,
[arXiv 2504.03955](https://arxiv.org/abs/2504.03955)): riproduzione del caso "surface power", studio del training set,
ed estensione a un chip a **due materiali** (conducibilità a gradino) confrontata con soluzioni Ansys.

Diari completi con figure: [`report_esperimenti.pdf`](report_esperimenti.pdf) (parte 1, caso surface del paper) e [`report_due_materiali.pdf`](report_due_materiali.pdf) (parte 2, chip a due materiali).
Contesto tecnico e tabella di tutti i run: [`CLAUDE.md`](CLAUDE.md).

## Contenuto

| cartella / file | cosa |
|---|---|
| `heat_surface.py`, `heat_volumetric.py` | script originali con fix di compatibilità jax/optax e nuovi flag (`--decay_steps`, `--train_data`, `--tag`) |
| `models.py`, `kan.py`, `hvp.py`, `train.py`, `eval.py` | copie **non modificate** dalla repo originale, necessarie per eseguire |
| `gen_block_maps.py` | generatore di mappe di potenza a blocchi e training set misto GRF + blocchi |
| `new_experiment/heat_surface3.py` | training a due materiali: loss PINN o **discretizzata FD** (stile DeepOHeat-v2), BC parametriche, `--pin_level`, augmentation, eval multi-caso |
| `new_experiment/models_kink.py` | `DeepOHeat_kink`: trunk in y con feature `\|y − y_i\|` per rappresentare il kink all'interfaccia |
| `new_experiment/check_truth.py` | verifica alle differenze finite che la ground truth Ansys soddisfi la loss (da lanciare prima di allenare) |
| `new_experiment/heat_surface2.py`, `models1.py`, `rescale.py`, `Preliminar_experiment.pdf` | materiale di partenza dell'esperimento a due materiali (F. Spinelli), tenuto come riferimento, non modificato |
| `scripts/` | script per figure e report |
| `data/` | dati Ansys (casi 11 e 14) e mappe di test derivate; i dati del paper vanno scaricati (vedi `data/README.md`) |
| `results/` | metriche, log, figure e pesi di ogni run (vedi `results/README.md`) |

## Installazione

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
Testato su macOS, solo CPU (jax 0.11, equinox 0.13, optax 0.2). `cupy` non è installabile su Mac: il notebook ibrido
della repo originale non è incluso.

## Riprodurre — parte 1: caso surface del paper

Scaricare `fs_train_surface.npy`, `fs_test_surface.npy`, `u_test_surface.npy` in `data/` (link in `data/README.md`).

```bash
# 1. default della repo (= Tabella I del paper): MAPE 0.058%, rel L2 2.96%, ~70 s su CPU
python heat_surface.py
# 2. decadimento del learning rate più lento: rel L2 1.96%
python heat_surface.py --epochs 50000 --decay_steps 1000
# 3. training set misto GRF + blocchi (miglior risultato: rel L2 1.10%, MAPE 0.031%)
python gen_block_maps.py --pmax 4 --seed 2 --out data/fs_train_surface_mixed_p4.npy
python heat_surface.py --epochs 50000 --decay_steps 1000 --train_data data/fs_train_surface_mixed_p4.npy --tag _mixed_p4
```
Ogni run scrive in `results/results_surface/DeepOHeat_v1/<architettura><tag>/`.

## Riprodurre — parte 2: chip a due materiali (k = 1.4 / 0.5, interfaccia a y ≈ 0.48)

```bash
# 0. la ground truth deve soddisfare la loss: residui ~0
python new_experiment/check_truth.py --u data/u_test_11.npy --f data/fs_test_11_flux.npy
# 1. sanity check della pipeline sul caso del paper (k1 = k2 = 1)
python new_experiment/heat_surface3.py --model kink --k1 1 --k2 1 --bottom_b 0.2 \
    --test_f data/fs_test_surface.npy --test_u data/u_test_surface.npy --epochs 10000 --tag _stepA
# 2. modello finale: loss FD + kink + livello fissato (caso 11: rel L2 0.7%, max err 0.22 °C; caso 14: 1.3%, 0.47 °C)
python new_experiment/heat_surface3.py --model kink --loss fd --fd_nx 41 --fd_nz 21 --lam_energy 1 \
    --amp_aug_min 0.1 --pin_level 1 --epochs 30000 --decay_steps 600 --tag _fd41_aug_pin \
    --test_f data/fs_test_11_flux.npy,data/fs_test_14_flux.npy --test_u data/u_test_11.npy,data/u_test_14.npy
```
Il training set è lo stesso `fs_train_surface_mixed_p4.npy` della parte 1. Per il confronto con la PINN a collocazione
(che fallisce, vedi `results/two_materials/B*`): `--loss pinn` con/senza `--z_random 1`.

## Risultati in breve

**Surface (paper)** — rel L2 medio sulle 10 mappe di test: default 2.96% → decay lento 1.96% → training misto 1.10%.
Il MAPE in Kelvin del paper (0.049%) nasconde errori di ~2 °C sui picchi. Il training solo a blocchi fallisce (4.4%):
i campi gaussiani servono.

**Due materiali** — la loss di partenza descriveva un problema diverso da quello Ansys (potenza ×5, coefficiente di
fondo ×10); la PINN a collocazione "bara" tra i punti (strato limite finto / Laplaciano oscillante) e sbaglia il livello
di 2 °C; la loss discretizzata FD risolve (33% → 1%). Il livello residuo è deterministico e dipende dal flusso della
mappa di input: `--pin_level` lo fissa a `a + b·⟨f⟩`. Dettagli e tabella in `CLAUDE.md`.

Avvertenze: un solo seed per configurazione; le mappe di test hanno guidato il disegno del generatore di blocchi;
i coefficienti fisici del caso Ansys (f = 0.2 per q = 2500 W/m², b = 2.0, T_amb = 25 °C) sono **misurati dai dati**,
non presi dal setup Ansys.
