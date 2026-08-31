# DeepOHeat-v1 — contesto di progetto

Clone di https://github.com/xlyu0127/DeepOHeat-v1, codice del paper *DeepOHeat-v1: Efficient Operator Learning for Fast and
Trustworthy Thermal Simulation and Optimization in 3D-IC Design* (Yu et al., UCSB, IEEE TCPMT 2025, arXiv 2504.03955).
Copia locale del paper: `paper.pdf`. Diario completo degli esperimenti con figure: `report_esperimenti.pdf`.

## Cosa fa
Rete operatoriale physics-informed: input = mappa di potenza 21×21 sulla superficie superiore di un chip
(1×1×0.5 normalizzato), output = temperatura 3D (valutata su 101×101×51). Non usa soluzioni di training: la loss
impone l'equazione di Laplace nel volume, `uz = f` sul top, convezione sul fondo, lati isolati. Temperatura reale
= `25*u + 293.15` K. Struttura "separabile" (einsum `iyz,jyz,kyz,byz->bijky`, rango r=128): per questo un training
completo dura ~1 min su CPU.

## Ambiente
- `.venv` con Python 3.12 (Homebrew; jax non ha wheel affidabili per 3.14). `source .venv/bin/activate`.
- `requirements.txt` scritto da noi (la repo non lo aveva). `cupy` non installabile su Mac → `hybrid_solver.ipynb` non gira.
- Solo CPU: ~7 ms/iterazione → 70 s per 10k epoche, ~6.5 min per 50k.
- `data/` (da Google Drive, gitignored) contiene solo mappe di input e soluzioni di riferimento del test. Nessun peso
  pre-allenato nella repo: tutti i modelli in `results/` li abbiamo allenati noi.

## Modifiche fatte al codice (non committate al 29/08/2026)
- `heat_surface.py`, `heat_volumetric.py`: `jax.tree_map` → `jax.tree_util.tree_map`;
  `optimizer.init(eqx.filter(model, eqx.is_inexact_array))` (senza questo optax recente crasha con "pytree structure error").
- `heat_surface.py`: flag `--decay_steps` (default 500), `--train_data` (default `data/fs_train_surface.npy`),
  `--tag` (suffisso della cartella risultati). Default = comportamento originale.
- `gen_block_maps.py` (nuovo): genera mappe a blocchi rettangolari e training set misto GRF+blocchi
  (`--n_blocks --n_grf --pmax --seed --out`).
- `.gitignore`: aggiunto `.venv/`.
- `heat_volumetric.py` NON è mai stato eseguito (paper: 100k iterazioni, 0.5 h su GPU).

## Esperimenti (caso surface, test = 10 mappe a blocchi con riferimento Celsius 3D)
Cartelle in `results/results_surface/DeepOHeat_v1/nf50_nc21_branch_8_256_trunk_3_64_r128<suffisso>/`, ognuna con
pesi `.eqx`, `u_pred_heat3d.npy`, log e figure.

| suffisso | setup | MAPE | rel L2 | note |
|---|---|---|---|---|
| paper | 10k ep, decay 500, RTX 3090 | 0.049% | — | Tabella I del paper |
| `_ep10000` | default repo | 0.058% | 2.96% | riproduce il paper; picchi sfocati (−2 °C) |
| `_ep50000` | 50k ep, decay 500 | — | 2.04% | loss piatta da 20k: lr → 5e-6 |
| `_ep50000_d1000` | 50k ep, decay 1000 | 0.043% | 1.96% | loss 4× più bassa, test quasi uguale |
| `_mixed` | + train 50% GRF + 50% blocchi (pmax 2) | 0.028% | 1.14% | migliora 8/10 mappe |
| `_blocks_p4` | train 100% blocchi (pmax 4) | 0.110% | 4.42% | PEGGIORE di tutti: allucina blob |
| `_mixed_p4` | train 50/50, pmax 4 | 0.031% | 1.10% | **migliore sui picchi** (0.27 °C medio) |

Comando del modello migliore:
`python gen_block_maps.py --pmax 4 --seed 2 --out data/fs_train_surface_mixed_p4.npy &&
python heat_surface.py --epochs 50000 --decay_steps 1000 --train_data data/fs_train_surface_mixed_p4.npy --tag _mixed_p4`

## Cose capite (non ovvie dal codice)
- Il MAPE in Kelvin mediato sul volume nasconde errori locali di ~2 °C sui picchi; guardare `max_l1` e le figure.
- Con lo scheduler originale (decay ×0.9 ogni 500) il training si spegne a ~20-25k epoche; oltre non serve.
- Train set del paper = campi gaussiani lisci (valori −4.5..4.6); test = blocchi netti 0/0.25/0.5/1 (mappa 9 arriva a 4).
  Il paper lo ammette (pag. 9). La sfocatura è in buona parte distribution shift.
- I GRF però SERVONO: senza (100% blocchi) il modello memorizza template e allucina. Misto batte entrambi i puri.
- La soluzione vera è essa stessa smussata (transizione ~0.2 di dominio al bordo di un blocco, in superficie);
  l'obiettivo è la giusta quantità di smussamento, non spigoli netti.
- Il generatore di blocchi è stato disegnato guardando le 10 mappe di test → dimostrazione di meccanismo, non confronto
  alla pari col paper. Manca un test set indipendente (servirebbe un solutore numerico; GMRES del notebook richiede CUDA).
- Un solo seed per run: differenze < 0.1% di rel L2 sono rumore.

## Convenzioni
- Prima di rilanciare un run, rinominare/copiare la cartella risultati o usare `--tag`: la cartella dipende solo
  dall'architettura e verrebbe sovrascritta.
- Run lunghi: in background con log su file + monitor filtrato (una riga ogni 10k epoche), non ogni 100 (troppo output).
- Non committare `data/`, `results/`, `.venv/` (già in `.gitignore`).

## Nuovo esperimento: chip a due materiali (`new_experiment/`, da 29/08/2026)
Problema (PDF `Preliminar_experiment.pdf`, F. Spinelli): k = 1.4 per y < ~0.48, k = 0.5 oltre; ground truth da Ansys
(`temperatura_matrice_11.npy` = rettangolo di potenza `[8:12,5:16]` di `rescale.py`; `_14` = quadrante nell'angolo, altro caso).
File dell'autore: `heat_surface2.py`, `models1.py` (DeepOHeat_phi), `rescale.py` — NON modificarli, servono da riferimento.

Diagnosi (`new_experiment/check_truth.py`, differenze finite sulla ground truth — lanciarlo prima di ogni training):
- la loss di `heat_surface2.py` descrive un problema diverso da quello Ansys: potenza f=1 dove i dati danno k*uz = 0.2
  (q = 2500 W/m2 -> 0.2 nelle unita' del codice, non 1); BC fondo `u - 0.2 = 0.2*k*uz` dove i dati danno coefficiente 2.0
  (T_amb = 25 °C, verificato su entrambi i casi 11 e 14 con bilancio energetico); interfaccia a y≈0.48 (codice: 0.45, phi: 0.5).
- la soluzione ha un kink (uy salta di k1/k2 = 2.8) che il trunk ChebyKAN liscio non puo' rappresentare; il termine
  `k_mean*lap = 0` all'interfaccia e' fisicamente sbagliato (serve continuita' del flusso k*uy).
- il dataset misto GRF+blocchi NON era il problema.

Codice nuovo: `new_experiment/models_kink.py` (DeepOHeat_kink: trunk y con [y, |y-yi|]; `use_kink=False` = baseline),
`new_experiment/heat_surface3.py` (loss corretta e parametrica, log dei termini, eval con figura). Lanciare dalla root.
Test set coerente: `data/fs_test_11.npy` (0.2*rettangolo), `data/u_test_11.npy`. Risultati in `results/results_2mat/<model><tag>/`.
Step A (k1=k2=1, caso paper, 10k ep): rel L2 3.5% -> pipeline ok.

Risultati (29/08 sera), tutti su caso 11, modello kink, train `fs_train_surface_mixed_p4.npy`:
| run | loss | rel L2 | note |
|---|---|---|---|
| `kink_case11_zfixed_CHEAT` | PINN, z fissa | 33% | strato limite finto al top (uzz 260x nell'ultimo 0.05): BC top soddisfatta, PDE ignorata tra i punti |
| `kink_case11` | PINN, z casuale | 30% | forma giusta ma livello -2 °C: Laplaciano oscilla tra i punti (RMS 1.0 su griglia fine), bilancio energetico rotto |
| `v1_case11` | PINN, z casuale, senza |y-yi| | 24% | termine interfaccia 300x peggio del kink |
| **`kink_case11_fd`** | **FD discretizzata (v2-style) 41x41x21 + bilancio energetico, 30k ep, decay 600** | **1.34%** | picco -0.27 °C, max err 0.41 °C, bilancio chiuso; caso 14 (mai visto) 5.9%, picco -0.9 °C |
Comando: `python new_experiment/heat_surface3.py --model kink --loss fd --fd_nx 41 --fd_nz 21 --lam_energy 1 --epochs 30000 --decay_steps 600 --tag _case11_fd`
Lezione: con b=2.0 e f=0.2 il livello di T e' 50x piu' sensibile agli errori di flusso che nel paper; la PINN con
collocazione non regge, la loss discretizzata si'. `--nz/--z_random` restano per la PINN. `eval.txt` riporta il
"boundary-layer check" (>10 = il modello bara). Test set caso 14: `data/fs_test_14.npy` (ricavato da k*uz), `data/u_test_14.npy`.
Attenzione input di test: il rettangolo di rescale.py (44 celle x 0.2) immette il 16% di calore in meno del riscaldatore Ansys
(flusso vero <k*uz> = 0.0230 vs 0.0200): parte dell'offset negativo dei run viene da qui. `data/fs_test_11_data.npy` e' la mappa
ricavata dai dati (media per cella, 65 celle): su di essa `kink_case11_fd` da' rel L2 0.47%, picco -0.21 °C.
Verifica utile: <u_fondo> deve valere a + b*<f> (bilancio energetico esatto); flag `--pin_level 1` lo impone come vincolo duro.
`--amp_aug_min 0.1` scala meta' del batch a ampiezze piccole; `--test_f/--test_u` accettano liste separate da virgola.
Errore residuo di forma senza offset: 0.4-0.9% in tutti i run FD; il livello (uno scalare per mappa) e' il termine dominante.
Serie D (29/08 sera): `kink_fd41_aug` (aug ampiezza 0.1-1, 50k ep) e `kink_fd41_aug_pin` (+ `--pin_level 1`, 30k ep):
forma 0.3-0.7% su tutti i casi come il primo run FD; il livello residuo con pin_level e' DETERMINISTICO = 25*b*(<f_input> - <f_vero>),
cioe' dipende solo da quanto la mappa 21x21 sottostima il flusso Ansys (celle di bordo a mezza area pesate come intere: -8%).
Mappe con flusso totale esatto: `data/fs_test_{11,14}_flux.npy` (media 21 punti = <k*uz> vero). D2 (griglia 51) interrotto: non necessario.
Regola pratica: per un nuovo caso Ansys, ricavare f da k*uz del top e scalare la mappa 21x21 al flusso totale vero; usare pin_level.

## Esperimento k trainable: k come input della rete (`new_experiment/`, 31/08/2026)
Problema (PDF `k_trainable.pdf`, F. Spinelli): stesso chip a due materiali, ma il campo k non e' piu' noto al modello
a priori: entra come input tramite un secondo branch (prodotto di Hadamard con il branch della potenza, Fig. 1 del PDF).
File nuovi: `new_experiment/models_k.py` (DeepOHeat_k: branch k su profilo k(y) a 21 punti pesato per frazione di volume,
input in log; trunk y con dizionario di kink opzionale), `new_experiment/heat_surface4.py` (loss FD con k per campione,
(k1,k2,yi) campionati a ogni batch: `--k_mode fixed|yi|full`, k in [0.3,2] loguniforme, yi in [0.2,0.8]),
`new_experiment/fd_solver.py` (solutore diretto AMG+CG, ~2 s per caso 101x101x51; validato vs Ansys: 0.24%/0.94% con f esatta),
`new_experiment/make_k_testsets.py` (test .npz: f,u,k1,k2,yi; `ktest_ansys11/14.npz` + `ktest_var.npz`, 12 casi con 6 k mai visti).

Risultati (31/08), train `fs_train_surface_mixed_p4.npy`, FD 41x41x21, pin_level, amp aug, cartelle `results/results_ktrain/`:
| run | Ansys 11 | Ansys 14 | ktest_var (12 k mai visti) |
|---|---|---|---|
| `kfixed_sanity` (k fisso, 30k) | 0.65% | 1.36% | 2.55% (k fuori distr.) |
| `kyi_30k` (solo yi variabile) | 1.06% | 5.39% | 2.26% |
| `kfull_50k` | 1.20% | 5.55% | 1.69% |
| `kfull_nokink` (30k, senza dizionario) | 1.24% | 5.61% | 1.64% |
| **`kfull_60k_d1200`** (decay 1200, kbranch 128) | **1.15%** | **5.20%** | **1.56%** (max err medio 0.87 °C) |

Cose capite:
- Sanity ok: il secondo branch non costa nulla a parita' di problema (0.65/1.36 vs 0.72/1.33 del preliminare).
- Il DIZIONARIO di kink [y,|y-c_m|] e' INUTILE con la loss FD (nokink identico): il trunk liscio basta; era la PINN
  a rendere indispensabile la feature |y-yi|. Ipotesi architetturale smentita dai dati.
- Tassa di condizionamento: lo stesso k, servito dal modello k-variabile, costa ~2x di rel L2 (0.9->2.0% sui casi FD).
  Sul caso Ansys 14 (quadrante, mappa piu' fuori distribuzione) costa 4x: errore = TILT lungo y (+-0.7 °C ai lati
  dell'interfaccia, ripartizione del salto termico leggermente sbagliata), NON offset (pin_level resta esatto per ogni k).
- decay_steps 600 spegne il training a ~30k epoche (lr 5e-6): per run >30k usare decay ~1200. Guadagno pero' modesto.
- Se a inferenza k e' noto, un fine-tune breve a k fisso recupera la precisione del modello dedicato (~0.7-1.4%).
Comando del run migliore: `python new_experiment/heat_surface4.py --k_mode full --epochs 60000 --decay_steps 1200 --kbranch_hidden 128 --tag _60k_d1200`

Serie accorgimenti (31/08 pomeriggio), tutti k_mode full senza dizionario, loss con interfaccia sub-griglia
(facce y = armonica pesata theta, nodi = media di volume; stessa cosa in fd_solver.solve_k_step):
| run | Ansys 11 | Ansys 14 | ktest_var |
|---|---|---|---|
| `kfull_subgrid` (30k) | 1.26% | 5.55% | 1.69% |
| `kfull_subgrid_s1` (seed 1) | 1.31% | 5.64% | 1.71% |
| `kfixed_ft_from_subgrid` (fine-tune k fisso, 10k ep, lr 3e-4, 107 s) | **0.60%** | **1.35%** | — |
- Interfaccia sub-griglia: NON era la causa del tilt del caso 14 (numeri identici alla loss quantizzata).
  Tenuta comunque nel codice: fisica piu' corretta, costo zero.
- Rumore da seed: ~0.05-0.1% di rel L2 -> il guadagno del run 60k/decay1200 (1.56% vs 1.69%) e' reale ma marginale.
- FINE-TUNE a k fisso dal modello generale (`--init_from`): recupera TUTTA la tassa di condizionamento in ~2 min
  (0.60/1.35 vs 0.65/1.36 del modello dedicato). Ricetta consigliata: modello generale per esplorare k,
  fine-tune quando k e' fissato. Inseguire oltre il tilt del caso 14 (capacita', famiglia f) ha ROI basso.
