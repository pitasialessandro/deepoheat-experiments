# Dati

## Inclusi
- `temperatura_matrice_11.npy`, `temperatura_matrice_14.npy` — soluzioni Ansys (101×101×51, °C) del chip a due materiali.
  Caso 11: riscaldatore rettangolare centrale; caso 14: quadrante nell'angolo.
- `u_test_11.npy`, `u_test_14.npy` — le stesse normalizzate come nel codice: `u = (T_°C − 20) / 25`, shape (1,101,101,51).
- Mappe di potenza 21×21 (shape (1,441)) per il test:
  - `fs_test_11.npy` — rettangolo `[8:12, 5:16]` di `rescale.py` × 0.2 (16% di calore in meno del vero);
  - `fs_test_11_data.npy`, `fs_test_14.npy` — ricavate da `k·∂u/∂z` sulla superficie top, con soglia;
  - `fs_test_*_cons.npy` — media esatta per cella sensore;
  - `fs_test_*_flux.npy` — **da usare**: scalate in modo che la media a 21 punti coincida con il flusso totale vero.
  - `fs_test_paper_like.npy` — output di `rescale.py` (rettangolo × 1.0, scala sbagliata), solo per riferimento.

## Da scaricare (dati del paper, ~1.2 GB con il caso volumetrico)
Google Drive degli autori: https://drive.google.com/drive/folders/13g2dkNU1AU0OPGRPvkBAncguAK7Cb6Ek?usp=sharing
Servono qui: `fs_train_surface.npy` (10000×21×21, campi gaussiani), `fs_test_surface.npy`, `u_test_surface.npy`.

## Generati
`python gen_block_maps.py --pmax 4 --seed 2 --out data/fs_train_surface_mixed_p4.npy` ricrea il training set misto
usato in tutti i run (5000 GRF + 5000 mappe a blocchi).
