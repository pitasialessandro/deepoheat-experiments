from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, KeepTogether
from reportlab.lib.utils import ImageReader

R = 'results/results_ktrain'
OUT = 'report_k_trainable.pdf'
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

s = [Paragraph('DeepOHeat con k in input', TITLE),
     Paragraph('Diario degli esperimenti, parte 3: un modello per tutte le configurazioni di conduttività. 31 agosto 2026.', SUB)]

s.append(Paragraph('Il problema in due parole', H1))
s.append(Paragraph(
    'Stesso chip a due materiali della parte 2, ma con una differenza di fondo (PDF ' + C('k_trainable.pdf') + '): nella parte 2 il campo di '
    'conduttività era <b>noto a priori</b> e il modello era costruito su quell\'informazione (k1, k2 e l\'interfaccia dentro la loss, la feature '
    '|y − y_i| nel trunk). Adesso il campo k <b>non è più cablato nel modello</b>: entra come input, tramite un secondo branch il cui output '
    'moltiplica (prodotto di Hadamard) quello del branch della potenza, come nello schema del PDF. Un solo modello deve servire tutte le '
    'configurazioni: materiali diversi (k in 0,3–2) e interfaccia in posizione diversa (y_i in 0,2–0,8).', P))
s.append(Paragraph(
    'Il training resta physics-informed: nessuna soluzione di training, a ogni batch si campionano (k1, k2, y_i) e la loss FD conservativa '
    'della parte 2 usa il k del campione (media armonica sulle facce y). Tutto ciò che aveva funzionato è ereditato intatto: '
    + C('pin_level') + ' (la media del fondo è a + b·&lt;f&gt; per bilancio energetico, <b>indipendente da k</b>: il vincolo resta esatto per '
    'qualsiasi configurazione), augmentation di ampiezza, input a flusso esatto, boundary-layer check.', P))

s.append(Paragraph('Passo 0 — Un riferimento per k arbitrari: il solutore FD', H1))
s.append(Paragraph(
    'Con k in input il modello va validato su k <i>diversi</i>, ma la ground truth Ansys esiste per una sola configurazione (i casi 11 e 14). '
    'Era la prima "direzione con margine" della parte 2: un solutore diretto nostro. ' + C('new_experiment/fd_solver.py') + ' risolve '
    'div(k grad u) = 0 con le stesse BC della loss (stencil conservativo ai nodi, pesi di volume ai bordi, matrice simmetrica, AMG + CG): '
    '<b>~2 secondi per caso</b> a 101×101×51. Validazione contro Ansys:', P))
s.append(table([['caso', 'input', 'rel L2 vs Ansys', 'max err', 'offset medio'],
                ['11 (rettangolo)', 'f esatta (k·uz dai dati)', '0,24%', '0,08 °C', '−0,01 °C'],
                ['11', 'mappa 21×21 a flusso esatto', '1,51%', '0,28 °C', '+0,10 °C'],
                ['14 (quadrante)', 'f esatta (k·uz dai dati)', '0,94%', '0,22 °C', '−0,09 °C'],
                ['14', 'mappa 21×21 a flusso esatto', '0,70%', '0,23 °C', '−0,07 °C']],
               col_widths=[3.4*cm, 5.6*cm, 3.2*cm, 2.4*cm, 2.7*cm]))
s.append(Spacer(1, 4))
s.append(Paragraph(
    'Con f esatta il disaccordo è 0,1–0,2 °C: il livello degli artefatti del riferimento Ansys stesso (generato su 21×21×11 e interpolato). '
    'Test set (' + C('make_k_testsets.py') + '): i 2 casi Ansys più <b>12 casi con 6 configurazioni di k mai viste</b> (interfaccia a 0,30 e 0,72, '
    'materiali invertiti, k uniforme) su 2 mappe di potenza held-out.', P))

s.append(Paragraph('Il modello: un branch per k', H1))
s.append(Paragraph(
    C('new_experiment/models_k.py') + ' — il branch k è un MLP piccolo (21→64→…→r) che riceve il profilo k(y) sui 21 punti sensore, in scala '
    'logaritmica; nella cella a cavallo dell\'interfaccia il valore è la <b>media pesata per la frazione di volume</b>, così y_i è ricostruibile '
    'esattamente dall\'input anche tra i sensori (lezione della parte 2: l\'input conta quanto il modello). Il suo output moltiplica elemento per '
    'elemento quello del branch della potenza, e l\'einsum separabile resta identico.', P))
s.append(Paragraph(
    'Per il gomito, che ora può stare ovunque, l\'idea di partenza era un <b>dizionario di kink</b>: trunk y con [y, |y−c_1|, …, |y−c_17|] su una '
    'griglia fissa di centri, con il branch k a pesare le componenti. Spoiler della sezione risultati: <b>non serve</b>.', P))

s.append(Paragraph('Risultati', H1))
s.append(table([['run', 'Ansys 11', 'Ansys 14', '12 k mai visti', 'note'],
                ['k fisso (sanity, 30k ep)', '0,65%', '1,36%', '2,55%', 'k fuori distribuzione nel 3° set'],
                ['solo y_i variabile (30k)', '1,06%', '5,39%', '2,26%', ''],
                ['famiglia completa (50k)', '1,20%', '5,55%', '1,69%', ''],
                ['completa senza dizionario (30k)', '1,24%', '5,61%', '1,64%', 'identico: dizionario inutile'],
                ['completa, 60k ep, decay 1200, branch k 128', '1,15%', '5,20%', '1,56%', 'max err medio 0,87 °C']],
               col_widths=[6.1*cm, 1.9*cm, 1.9*cm, 2.6*cm, 4.8*cm], bold_row=5))
s.append(Spacer(1, 6))
s += bullets(
    '<b>Sanity superata</b>: a parità di problema (k fisso) il modello a due branch rifà i numeri della parte 2 (0,65/1,36 contro 0,72/1,33). '
    'Il secondo branch non costa nulla.',
    '<b>La generalizzazione su k funziona</b>: 1,56% medio su 12 configurazioni mai viste (il modello a k fisso, sugli stessi casi, fa 2,55%). '
    'Interfacce a 0,30 e 0,72, materiali invertiti e k uniforme: forma corretta ovunque, casi facili sotto l\'1%.',
    '<b>Il dizionario di kink è inutile con la loss FD</b>: l\'ablation senza dà numeri identici con metà epoche. Il trunk liscio approssima il '
    'gomito da solo; era la PINN a collocazione a rendere indispensabile la feature |y−y_i| (300× sul termine d\'interfaccia). I run successivi '
    'girano senza: modello più semplice.')
s.append(fig(f'{R}/kfull_60k_d1200/eval_ktest_var/eval_worst.png',
             caption='Fig. 1 — Modello finale sul caso peggiore dei 12 mai visti (k = 0,7/0,35, interfaccia a y = 0,72): il gomito è al posto '
                     'giusto anche dove il modello non ha mai visto un\'interfaccia in quel punto con quel contrasto.'))

s.append(Paragraph('La tassa di condizionamento e il caso 14', H1))
s.append(Paragraph(
    'Il prezzo del modello unico: sullo <i>stesso</i> k della parte 2, il rel L2 raddoppia (0,9 → 2,0% sui casi FD) e sul caso Ansys 14 quadruplica '
    '(1,36 → 5,2%). La diagnosi esclude il livello: ' + C('pin_level') + ' tiene (offset −0,13 °C, uguale al modello dedicato). L\'errore è un '
    '<b>tilt lungo y</b>: −0,8 °C sul lato k1, +0,7 °C sul lato k2, zero vicino all\'interfaccia — il modello che serve tutti i k ripartisce '
    'leggermente male il salto termico tra i due materiali, e paga di più sulla mappa più atipica (il quadrante appoggiato alle pareti).', P))
s.append(fig(f'{R}/diagnostica_tilt_case14.png',
             caption='Fig. 2 — Caso 14. Sinistra: errore medio lungo y — il modello a k fisso (blu) è piatto, il modello generale (rosso) ha il tilt '
                     'a cavallo dell\'interfaccia; il fine-tune (verde) lo azzera. Destra: profilo di temperatura in superficie.'))
s.append(Paragraph(
    'Due ipotesi verificate e <b>smentite</b>: (1) la quantizzazione dell\'interfaccia sulla griglia FD della loss (passo 0,025, bias variabile da '
    'campione a campione) — corretta con la media armonica pesata per la frazione di segmento sulla faccia a cavallo: numeri identici (5,55 vs 5,61%); '
    'la modifica resta perché è fisica più corretta a costo zero. (2) Il learning rate: con decay 600 il training muore a ~30k epoche; a 60k epoche con '
    'decay 1200 il guadagno c\'è ma è marginale (1,69 → 1,56%). Un secondo seed misura il rumore: ~0,1% di rel L2.', P))

s.append(Paragraph('Il rimedio che funziona: fine-tune a k fisso', H1))
s.append(Paragraph(
    'Se a inferenza la configurazione è nota, 10k epoche a k fisso (lr 3e-4) partendo dai pesi del modello generale — flag ' + C('--init_from') + ' — '
    'recuperano <b>tutta</b> la tassa di condizionamento in <b>107 secondi</b>:', P))
s.append(table([['modello', 'Ansys 11', 'Ansys 14', 'costo'],
                ['dedicato, allenato da zero (parte 2)', '0,72%', '1,33%', '~6 min'],
                ['generale (k in input)', '1,15%', '5,20%', '—'],
                ['generale + fine-tune 10k ep', '0,60%', '1,35%', '107 s']],
               col_widths=[7.2*cm, 2.4*cm, 2.4*cm, 2.4*cm], bold_row=3))
s.append(Spacer(1, 4))
s.append(Paragraph(
    'Flusso di lavoro consigliato: il modello generale per esplorare materiali e posizioni dell\'interfaccia, il fine-tune lampo quando la '
    'configurazione è decisa. Inseguire il tilt residuo con più capacità o famiglie di mappe ad hoc ha ROI basso: sotto ~0,5 °C si fittano gli '
    'artefatti del riferimento (lezione della parte 2).', P))
s.append(fig(f'{R}/kfixed_ft_from_subgrid/eval_ktest_ansys14/eval_worst.png',
             caption='Fig. 3 — Caso 14 dopo il fine-tune: rel L2 1,35%, errore massimo 0,44 °C. Il tilt della Fig. 2 è sparito.'))

s.append(Paragraph('Cosa abbiamo imparato', H1))
s += bullets(
    'Lo schema del PDF (secondo branch + Hadamard) funziona così com\'è: la parte difficile non era l\'architettura ma <b>avere un riferimento</b> '
    'per k arbitrari (il solutore) e capire dove finisce la capacità del modello unico (il tilt).',
    'La loss FD conservativa generalizza a k variabile quasi gratis: k per campione e media armonica sulle facce. Nessun termine d\'interfaccia ad hoc, '
    'per nessuna posizione dell\'interfaccia.',
    C('pin_level') + ' è indipendente da k: il livello resta fissato dalla fisica anche quando il materiale cambia a ogni campione.',
    'Il dizionario di kink era un\'ipotesi ragionevole e sbagliata: con la fisica definita sui punti FD, il trunk liscio basta. Vale il rasoio: '
    'feature in meno, modello più semplice.',
    'Un\'ora di GPU non serve: tutti i run di questo report sono CPU, 7–14 minuti l\'uno (11 ms/iterazione).',
    'Il fine-tune da modello generale è più veloce <i>e</i> leggermente migliore del modello dedicato da zero: il generale è un buon prior.')

s.append(Paragraph('Cosa NON dimostrano questi esperimenti', H2))
s += bullets(
    'I 12 casi di test sono generati con lo <b>stesso stencil</b> della loss (griglie diverse: 101 vs 41): un errore sistematico comune a entrambi non '
    'si vedrebbe. I casi Ansys restano l\'unico controllo esterno, e coprono un solo k.',
    'La famiglia è due materiali stratificati lungo y. Il branch k accetta qualsiasi profilo, ma più strati, gradienti continui o k(x,y,z) pieno '
    'non sono stati né allenati né testati.',
    'Le mappe f dei test a k variabile sono blocchi: il tilt del caso 14 suggerisce che la copertura di mappe "a parete" nel training set conti; '
    'non è stato isolato quanto.',
    'Un seed per configurazione (due per la migliore): differenze sotto ~0,1–0,3% sono rumore.')

s.append(Paragraph('Appendice — Riprodurre', H1))
s.append(Paragraph('Dalla root della repo, venv attivo. Training set: lo stesso misto GRF + blocchi delle parti 1–2.', P))
s.append(Paragraph(
    '# solutore e test set (valida contro Ansys, poi genera i .npz)<br/>'
    'python new_experiment/fd_solver.py --validate<br/>'
    'python new_experiment/make_k_testsets.py<br/>'
    '# modello generale (run migliore)<br/>'
    'python new_experiment/heat_surface4.py --k_mode full --kink_centers 0 --epochs 60000 \\<br/>'
    '&nbsp;&nbsp;--decay_steps 1200 --kbranch_hidden 128 --tag _60k<br/>'
    '# fine-tune a k fisso dal modello generale<br/>'
    'python new_experiment/heat_surface4.py --k_mode fixed --kink_centers 0 --epochs 10000 --decay_steps 600 \\<br/>'
    '&nbsp;&nbsp;--lr 3e-4 --init_from results/results_ktrain/kfull_60k/model.eqx \\<br/>'
    '&nbsp;&nbsp;--test data/ktest_ansys11.npz,data/ktest_ansys14.npz --tag _ft', CODE))
s.append(Paragraph('File nuovi: ' + C('new_experiment/models_k.py') + ', ' + C('new_experiment/heat_surface4.py') + ', '
                   + C('new_experiment/fd_solver.py') + ', ' + C('new_experiment/make_k_testsets.py') + '. '
                   'Risultati e pesi in ' + C('results/results_ktrain/') + '. Nota: il run del report è ' + C('kfull_60k_d1200')
                   + ' (dizionario di kink attivo ma ininfluente); il comando sopra riproduce la variante consigliata, senza dizionario.', P))

def footer(canvas, doc):
    canvas.saveState(); canvas.setFont('Helvetica', 8); canvas.setFillColor(colors.HexColor('#888888'))
    canvas.drawRightString(A4[0] - 1.8*cm, 1.1*cm, f'DeepOHeat — k in input — pag. {doc.page}'); canvas.restoreState()
doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=1.8*cm, rightMargin=1.8*cm, topMargin=1.8*cm, bottomMargin=1.8*cm,
                        title='DeepOHeat con k in input — diario esperimenti parte 3', author='Alessandro Pitasi')
doc.build(s, onFirstPage=footer, onLaterPages=footer); print(OUT)
