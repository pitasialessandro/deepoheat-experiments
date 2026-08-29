from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, KeepTogether
from reportlab.lib.utils import ImageReader

R = 'results/two_materials'
OUT = 'report_due_materiali.pdf'
ss = getSampleStyleSheet()
H1 = ParagraphStyle('H1', parent=ss['Heading1'], fontSize=17, spaceBefore=14, spaceAfter=6, textColor=colors.HexColor('#1f2d3d'))
H2 = ParagraphStyle('H2', parent=ss['Heading2'], fontSize=13, spaceBefore=10, spaceAfter=4, textColor=colors.HexColor('#2c3e50'))
P = ParagraphStyle('P', parent=ss['BodyText'], fontSize=10, leading=14, spaceAfter=6)
B = ParagraphStyle('B', parent=P, leftIndent=12, bulletIndent=2, spaceAfter=3)
CAP = ParagraphStyle('CAP', parent=P, fontSize=8.5, leading=11, textColor=colors.HexColor('#555555'), spaceAfter=10)
CODE = ParagraphStyle('CODE', parent=P, fontName='Courier', fontSize=8.5, leading=11, backColor=colors.HexColor('#f4f4f4'), borderPadding=4, leftIndent=4, spaceAfter=8)
TITLE = ParagraphStyle('T', parent=ss['Title'], fontSize=22, leading=27, spaceAfter=4)
SUB = ParagraphStyle('S', parent=P, fontSize=11, textColor=colors.HexColor('#666666'), spaceAfter=14)
W = A4[0] - 3.6 * cm

def fig(path, caption=None, width=W):
    iw, ih = ImageReader(path).getSize(); items = [Image(path, width=width, height=width * ih / iw)]
    if caption: items.append(Paragraph(caption, CAP))
    return KeepTogether(items)

CELL = ParagraphStyle('CELL', parent=P, fontSize=8.5, leading=10.5, spaceAfter=0)
CELLB = ParagraphStyle('CELLB', parent=CELL, fontName='Helvetica-Bold')

def table(rows, col_widths=None, bold_row=None):
    rows = [[Paragraph(str(c), CELLB if (i == 0 or i == bold_row) else CELL) for c in r] for i, r in enumerate(rows)]
    t = Table(rows, colWidths=col_widths, hAlign='LEFT')
    st = [('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e8eef4')), ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f7f9fb')]),
          ('GRID', (0, 0), (-1, -1), 0.3, colors.HexColor('#c8d0d8')), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
          ('TOPPADDING', (0, 0), (-1, -1), 3), ('BOTTOMPADDING', (0, 0), (-1, -1), 3)]
    if bold_row is not None:
        st += [('BACKGROUND', (0, bold_row), (-1, bold_row), colors.HexColor('#e6f4ea'))]
    t.setStyle(TableStyle(st)); return t

def bullets(*items): return [Paragraph(x, B, bulletText='•') for x in items]
C = lambda s: f'<font face="Courier">{s}</font>'

s = [Paragraph('DeepOHeat su un chip a due materiali', TITLE),
     Paragraph('Diario degli esperimenti, parte 2: dal "blob rosso" del primo tentativo a 0,2 °C di errore. 29 agosto 2026.', SUB)]

s.append(Paragraph('Il problema in due parole', H1))
s.append(Paragraph(
    'Stesso chip della parte 1 (1×1×0,5, riscaldatore sulla superficie superiore), ma fatto di <b>due materiali</b>: conducibilità '
    'k = 1,4 per y &lt; ~0,48 e k = 0,5 oltre. Un\'interfaccia netta, che nella temperatura produce un <i>gomito</i> (la pendenza cambia '
    'di un fattore 2,8 attraversandola). La soluzione di riferimento non viene più dal paper ma da <b>Ansys</b>, per due casi: '
    'il caso 11 (rettangolo di potenza al centro) e il caso 14 (quadrante nell\'angolo).', P))
s.append(Paragraph(
    'Punto di partenza: il codice dell\'autore del PDF (' + C('heat_surface2.py') + ', ' + C('models1.py') + ', ' + C('rescale.py') + ') '
    'estendeva la loss originale con k per regione e un termine d\'interfaccia. Risultato: un blob rosso saturo dove dovrebbe esserci un '
    'rettangolo tiepido, errore fino a 3 K, e il termine ' + C('loss_top') + ' che non scendeva mai. Epoche, batch, peso di loss_top, '
    'training set misto: niente aiutava. Domanda: <i>cosa fare?</i>', P))

s.append(Paragraph('Passo 0 — Prima di allenare: la soluzione vera soddisfa la loss?', H1))
s.append(Paragraph(
    'Controllo che vale sempre la pena fare e che qui ha deciso tutto. Con le differenze finite ho calcolato <b>ogni termine della loss '
    'direttamente sul campo Ansys</b>. Se la ground truth stessa "viola" la loss, la rete sta imparando un problema diverso da quello di cui '
    'abbiamo la soluzione, e nessun trucco di training può convergere.', P))
s.append(table([['termine', 'cosa imponeva il codice', 'cosa dice il dato Ansys', 'verdetto'],
                ['BC top  k·uz = f', 'f = 1 sul rettangolo', 'k·uz = 0,20 (e area un po\' più larga)', 'scala ×5 sbagliata'],
                ['BC fondo  u − 0,2 = b·k·uz', 'b = 0,2', 'b = 2,0 (bilancio energetico, entrambi i casi)', 'coefficiente ×10 sbagliato'],
                ['interfaccia', 'y = 0,45 (loss) / 0,5 (modello phi)', 'y ≈ 0,48; salto di uy = 2,83 ≈ k1/k2', 'posizione sbagliata'],
                ['Laplaciano nelle regioni', 'k·lap(u) = 0', 'RMS 0,01 vs 0,34 di uzz', 'ok']],
               col_widths=[4.2*cm, 4.2*cm, 5.3*cm, 3.6*cm]))
s.append(Spacer(1, 6))
s.append(fig(f'{R}/diagnostica/diagnostica_ansys_11.png',
             caption='Fig. 1 — Caso 11. Sinistra: temperatura Ansys in superficie. Centro: la potenza <i>ricavata dai dati</i> (k·uz al top) con sopra il '
                     'rettangolo scritto a mano in rescale.py: posizione giusta, ma il livello è 0,2, non 1. Destra: T lungo y con il gomito all\'interfaccia.'))
s.append(Paragraph(
    'Il rettangolo di ' + C('rescale.py') + ' era <b>scritto a mano</b> (la riga che converte il flusso Ansys era commentata) e la formula del paper '
    '"0,00625 mW per tile <-> 1" non vale per questo setup Ansys. Correzioni: f = 0,2·riscaldatore, b = 2,0, interfaccia 0,48. '
    'Dopo la correzione la ground truth dà residui ~0 (top 0,036 contro 0,277, fondo 0,006 contro 0,043). Script: ' + C('new_experiment/check_truth.py') + '.', P))

s.append(Paragraph('Passo 1 — Il gomito: una feature per il modello', H1))
s.append(Paragraph(
    'Il trunk in y è una ChebyKAN, cioè polinomi: liscia per costruzione, non può fare un gomito. Modifica minima (' + C('models_kink.py') + '): '
    'il trunk in y riceve <b>[y, |y − y_i|]</b>. La funzione |y − y_i| è continua con derivata discontinua: esattamente la forma della soluzione, e '
    'grazie alla struttura separabile basta darla al trunk in y. Il termine d\'interfaccia diventa la fisica giusta: <b>continuità del flusso</b> '
    'k1·uy(-) = k2·uy(+), non più "Laplaciano nullo sull\'interfaccia" (che chiede regolarità dove la fisica vuole una singolarità).', P))
s.append(fig(f'{R}/diagnostica/diagnostica_ansys.png',
             caption='Fig. 2 — Caso 14 (quadrante). Ultimo pannello: la derivata uy lungo y salta di un fattore ≈ 2,8 = k1/k2 a y ≈ 0,47: è il gomito che il modello deve saper fare.'))

s.append(Paragraph('Passo 2 — La PINN corretta: 33%. Il modello bara.', H1))
s.append(Paragraph(
    'Con loss corretta e feature del gomito, il primo run (B0) dà rel L2 = 33%: predizione piatta a 24 °C, <b>sotto la temperatura ambiente</b>, '
    'fisicamente impossibile con calore in ingresso. Eppure tutti i termini della loss sono ~1e-5 tranne top. Come si conciliano? '
    'Valutando il modello su una griglia fine in z si trova uno <b>strato limite finto</b>: nell\'ultimo centesimo di spessore uz schizza a 1/k '
    '(BC top soddisfatta), mentre sotto la temperatura è piatta. La PDE viene controllata solo sui punti di collocazione (z = 0,05, 0,10, …, 0,45): '
    'lo strato sta <i>tra</i> l\'ultimo punto e la superficie, dove nessuno guarda. Curvatura |uzz|: 260 volte più grande nell\'ultimo 0,05 che nel resto.', P))
s.append(fig(f'{R}/diagnostica/profilo_z_cheat_vs_fd.png',
             caption='Fig. 3 — Profilo T(z) sotto il riscaldatore, caso 11. Sinistra: il modello PINN (B0) è piatto e sotto l\'ambiente, con il gradiente '
                     'giusto solo nell\'ultimo punto. Destra: il modello finale (D3) segue la curva Ansys. I pallini sono i punti di collocazione della PINN.'))
s.append(Paragraph(
    'Rimedio classico: <b>punti di collocazione casuali</b> in z a ogni passo (run B1). Lo strato limite sparisce e la <i>forma</i> diventa giusta '
    '(il bump del rettangolo ha l\'ampiezza vera, il gomito c\'è), ma l\'intero campo è traslato di <b>−2 °C</b>: rel L2 30%.', P))
s.append(fig(f'{R}/B1_pinn_zrandom/eval_case0.png',
             caption='Fig. 4 — Run B1 (PINN, z casuale). Forma corretta, livello sbagliato di 2 °C: la predizione sta ancora sotto l\'ambiente.'))
s.append(Paragraph(
    'Diagnosi del livello: misurato su griglia fine, il Laplaciano del modello ha RMS <b>1,0</b> (contro 0,03 sui punti di collocazione): oscilla '
    'violentemente <i>tra</i> i punti, con media positiva. Integrato sul volume è un pozzo di calore: il flusso entra dal top (0,10) e non esce dal fondo '
    '(−0,006). Il livello di temperatura è fissato dalla BC di fondo, u − 0,2 = 2·k·uz: un errore di flusso dq nel flusso diventa dT = 25·2·dq nella temperatura. Con b = 2 e f = 0,2 '
    'il livello è <b>50 volte più sensibile</b> agli errori di flusso che nel paper (b = 0,2, f ≈ 1). Per questo il codice originale "funzionava" e qui no.', P))

s.append(Paragraph('Passo 3 — Loss discretizzata (stile DeepOHeat-v2): 1,3%', H1))
s.append(Paragraph(
    'Se il problema è "la fisica vale solo dove la si controlla", la risposta è definire la fisica <i>solo sui punti</i>: al posto delle derivate con '
    'autodiff, uno <b>stencil alle differenze finite</b> su una griglia densa (41×41×21), in forma conservativa. Tre vantaggi: non esiste più un "tra i punti" '
    'dove barare; l\'interfaccia si gestisce con la <b>media armonica di k</b> sulle facce, senza termini ad hoc; il bilancio energetico è conservativo per '
    'costruzione (l\'ho aggiunto anche come termine esplicito). Niente Hessiane → più veloce per punto. È la strada presa da DeepOHeat-v2.', P))
s.append(Paragraph('Prima di allenare, di nuovo il Passo 0: lo stencil FD sulla ground truth dà top 7e-4, fondo 3e-5, bilancio 4e-7. Ok.', P))
s.append(fig(f'{R}/D0_fd/eval_case0.png',
             caption='Fig. 5 — Run D0 (loss FD, 30k epoche). Caso 11: rel L2 1,34%, picco −0,27 °C, errore massimo 0,41 °C. Il gomito all\'interfaccia (profilo a destra) è riprodotto.'))
s.append(fig(f'{R}/D0_fd/eval_case14/eval_case0.png',
             caption='Fig. 6 — Stesso modello D0 sul caso 14, mai usato per costruire nulla: 5,9%. Forma giusta, ma tutto il campo ~0,6 °C sotto.'))

s.append(Paragraph('Passo 4 — Il livello: fisica esatta, e un input troppo debole', H1))
s.append(Paragraph(
    'In tutti i run FD l\'errore di <i>forma</i> (tolto l\'offset medio) è 0,3–0,9%; quello che domina è un <b>offset uniforme</b> di 0,1–0,6 °C, diverso '
    'da run a run. Ma il livello si può fissare senza chiederlo alla rete: per Laplace con lati adiabatici, la temperatura media del fondo è '
    '<b>esattamente</b> a + b·&lt;f&gt; — dipende solo dall\'input. Con ' + C('--pin_level') + ' aggiungo a ogni mappa la costante che impone questa media: '
    'una costante non tocca né la PDE né i flussi, solo lo scalare che la rete sbagliava.', P))
s.append(Paragraph(
    'Il vincolo ha funzionato al punto da smascherare l\'ultimo colpevole: l\'offset residuo era <b>deterministico</b> e pari a 25·b·(&lt;f_input&gt; − &lt;f_vero&gt;). '
    'Cioè: il rettangolo scritto a mano (44 celle × 0,2) immette il <b>16% di calore in meno</b> del riscaldatore Ansys, e anche la mappa ricavata dai dati '
    'ne perde un 8% (le celle di bordo, a mezza area, pesate come intere). Con mappe scalate al flusso totale vero (' + C('fs_test_*_flux.npy') + '):', P))
s.append(table([['run D3 (FD + kink + livello fissato), input a flusso esatto', 'rel L2', 'offset', 'picco', 'max errore', 'forma'],
                ['caso 11 — rettangolo centrale', '0,72%', '−0,04 °C', '−0,14 °C', '0,22 °C', '0,30%'],
                ['caso 14 — quadrante (mai visto)', '1,33%', '−0,13 °C', '−0,28 °C', '0,47 °C', '0,41%']],
               col_widths=[6.6*cm, 1.8*cm, 2*cm, 2*cm, 2.3*cm, 1.6*cm], bold_row=1))
s.append(Spacer(1, 6))
s.append(fig(f'{R}/D3_fd_aug_pin/eval_fs_test_11_flux/eval_case0.png',
             caption='Fig. 7 — Il risultato migliore: run D3, caso 11, input a flusso esatto. Scala dell\'errore ±0,2 °C (nella Fig. 4 era ±2,5 °C). '
                     'Resta una lieve sottostima ai bordi del riscaldatore: l\'input 21×21 non può rappresentare bordi che cadono tra i sensori.'))
s.append(fig(f'{R}/D3_fd_aug_pin/eval_fs_test_14_flux/eval_case0.png',
             caption='Fig. 8 — Run D3 sul caso 14 (mai visto): 1,33%, errore massimo 0,47 °C su un campo che varia di 4,3 °C.'))

s.append(Paragraph('Tutti i run in una tabella', H1))
s.append(table([['cartella', 'loss', 'modifiche', 'caso 11', 'caso 14'],
                ['(PDF di partenza)', 'PINN, k per regione', 'termine k_mean·lap(u)=0, f=1, b=0,2', '~3 K di errore', '—'],
                ['B0_pinn_zfixed_CHEAT', 'PINN', 'loss corretta + kink', '33%', '—'],
                ['B1_pinn_zrandom', 'PINN', '+ z casuale', '30%', '—'],
                ['C_pinn_v1_nokink', 'PINN', 'come B1 senza |y−y_i|', '24% (interfaccia 300× peggio)', '—'],
                ['D0_fd', 'FD 41×41×21', '+ bilancio energetico', '1,34%  (0,47% input da dati)', '5,9%'],
                ['D1_fd_aug', 'FD', '+ augmentation ampiezza, 50k ep', '3,0%', '3,0%'],
                ['D3_fd_aug_pin', 'FD', '+ livello fissato (pin_level)', '0,72% (input flusso esatto)', '1,33%']],
               col_widths=[3.9*cm, 2.3*cm, 4.6*cm, 3.9*cm, 2.3*cm], bold_row=7))
s.append(Spacer(1, 4))
s.append(Paragraph('Le percentuali sono rel L2 sull\'intero volume (101×101×51), che include la temperatura ambiente nel denominatore: un offset uniforme di 0,1 °C vale già ~1,3%. '
                   'Per l\'uso pratico guardare picco e massimo errore in °C.', P))

s.append(Paragraph('Cosa abbiamo imparato', H1))
s += bullets(
    'Prima di ogni training, verificare che la ground truth soddisfi la loss (' + C('check_truth.py') + '). Qui tre parametri su quattro erano sbagliati.',
    'Una PINN a collocazione può soddisfare tutte le BC e ignorare la PDE <i>tra</i> i punti. Con griglia fissa fa strati limite finti; con punti casuali fa oscillare il Laplaciano. Il rilevatore "boundary-layer check" in ' + C('eval.txt') + ' lo segnala.',
    'La sensibilità del livello agli errori di flusso è 25·b: con b = 2 il problema è 10× più severo che nel paper, e il segnale (f = 0,2) 5× più piccolo. Stesso codice, problema 50× più difficile.',
    'La loss discretizzata FD risolve alla radice: fisica definita solo sui punti, interfaccia con media armonica, conservazione dell\'energia. 33% → 1%.',
    'Il livello di temperatura è uno scalare noto dalla fisica (a + b·&lt;f&gt;): fissarlo è gratis e toglie il termine d\'errore dominante.',
    'L\'input conta quanto il modello: il rettangolo a mano immetteva il 16% di calore in meno. Con input fedele, D0 passa da 1,34% a 0,47% senza riallenare.',
    'La feature |y − y_i| è necessaria per la continuità del flusso all\'interfaccia (300× sul termine dedicato).',
    'Augmentation di ampiezza e griglia FD più fine: effetto nel rumore. Il run D2 (griglia 51) è stato interrotto.')

s.append(Paragraph('Cosa NON dimostrano questi esperimenti', H2))
s += bullets(
    'f = 0,2, b = 2,0, T_amb = 25 °C sono <b>misurati dai dati Ansys</b>, non presi dal setup: averli dal modello Ansys (q, h, dimensioni) chiuderebbe l\'offset residuo del caso 14.',
    'Due soli casi di test. Un solutore FD nostro (lo stencil c\'è già; ' + C('pyamg') + ' è installato) darebbe test set illimitati e permetterebbe di misurare la qualità del riferimento Ansys stesso, che è stato generato su 21×21×11 e poi interpolato.',
    'Un solo seed per configurazione: differenze sotto ~0,3% di rel L2 sono rumore.',
    'Sotto 0,2–0,5 °C di errore stiamo probabilmente fittando gli artefatti del riferimento: spingere ancora l\'accuratezza su questi due casi non è un buon investimento. Le direzioni con margine sono un solutore per i test, k come input del modello (un modello per molte configurazioni) e geometrie a più strati.')

s.append(Paragraph('Appendice — Riprodurre', H1))
s.append(Paragraph('Dalla root della repo, venv attivo. Training set: lo stesso misto GRF + blocchi della parte 1.', P))
s.append(Paragraph(
    '# verifica della loss sulla ground truth<br/>python new_experiment/check_truth.py --u data/u_test_11.npy --f data/fs_test_11_flux.npy<br/>'
    '# modello finale (D3), valutato sui due casi<br/>python new_experiment/heat_surface3.py --model kink --loss fd --fd_nx 41 --fd_nz 21 --lam_energy 1 \\<br/>'
    '&nbsp;&nbsp;--amp_aug_min 0.1 --pin_level 1 --epochs 30000 --decay_steps 600 --tag _fd41_aug_pin \\<br/>'
    '&nbsp;&nbsp;--test_f data/fs_test_11_flux.npy,data/fs_test_14_flux.npy --test_u data/u_test_11.npy,data/u_test_14.npy<br/>'
    '# controparte PINN che fallisce (B1): --loss pinn --z_random 1', CODE))
s.append(Paragraph('File: ' + C('new_experiment/heat_surface3.py') + ' (loss PINN/FD, BC parametriche, pin_level, augmentation, eval multi-caso), '
                   + C('new_experiment/models_kink.py') + ', ' + C('new_experiment/check_truth.py') + '. I file di partenza dell\'autore sono conservati intatti nella stessa cartella. '
                   'Risultati e pesi in ' + C('results/two_materials/') + '.', P))

def footer(canvas, doc):
    canvas.saveState(); canvas.setFont('Helvetica', 8); canvas.setFillColor(colors.HexColor('#888888'))
    canvas.drawRightString(A4[0] - 1.8*cm, 1.1*cm, f'DeepOHeat — due materiali — pag. {doc.page}'); canvas.restoreState()
doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=1.8*cm, rightMargin=1.8*cm, topMargin=1.8*cm, bottomMargin=1.8*cm,
                        title='DeepOHeat su un chip a due materiali — diario esperimenti', author='Alessandro Pitasi')
doc.build(s, onFirstPage=footer, onLaterPages=footer); print(OUT)
