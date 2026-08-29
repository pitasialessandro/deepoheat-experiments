from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
                                PageBreak, KeepTogether)
from reportlab.lib.utils import ImageReader

R = 'results/results_surface/DeepOHeat_v1/nf50_nc21_branch_8_256_trunk_3_64_r128'
OUT = 'report_esperimenti.pdf'

ss = getSampleStyleSheet()
H1 = ParagraphStyle('H1', parent=ss['Heading1'], fontSize=17, spaceBefore=14, spaceAfter=6, textColor=colors.HexColor('#1f2d3d'))
H2 = ParagraphStyle('H2', parent=ss['Heading2'], fontSize=13, spaceBefore=10, spaceAfter=4, textColor=colors.HexColor('#2c3e50'))
P = ParagraphStyle('P', parent=ss['BodyText'], fontSize=10, leading=14, spaceAfter=6)
B = ParagraphStyle('B', parent=P, leftIndent=12, bulletIndent=2, spaceAfter=3)
CAP = ParagraphStyle('CAP', parent=P, fontSize=8.5, leading=11, textColor=colors.HexColor('#555555'), spaceAfter=10)
CODE = ParagraphStyle('CODE', parent=P, fontName='Courier', fontSize=8.5, leading=11, backColor=colors.HexColor('#f4f4f4'),
                      borderPadding=4, leftIndent=4, spaceAfter=8)
TITLE = ParagraphStyle('T', parent=ss['Title'], fontSize=22, leading=27, spaceAfter=4)
SUB = ParagraphStyle('S', parent=P, fontSize=11, textColor=colors.HexColor('#666666'), spaceAfter=14)

W = A4[0] - 3.6*cm   # larghezza utile

def fig(path, width=W, caption=None):
    iw, ih = ImageReader(path).getSize()
    h = width * ih / iw
    items = [Image(path, width=width, height=h)]
    if caption: items.append(Paragraph(caption, CAP))
    return KeepTogether(items)

def table(rows, col_widths=None, bold_row=None):
    t = Table(rows, colWidths=col_widths, hAlign='LEFT')
    style = [('FONT', (0,0), (-1,0), 'Helvetica-Bold', 9), ('FONT', (0,1), (-1,-1), 'Helvetica', 9),
             ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#e8eef4')),
             ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f7f9fb')]),
             ('GRID', (0,0), (-1,-1), 0.3, colors.HexColor('#c8d0d8')),
             ('VALIGN', (0,0), (-1,-1), 'MIDDLE'), ('TOPPADDING', (0,0), (-1,-1), 3), ('BOTTOMPADDING', (0,0), (-1,-1), 3)]
    if bold_row is not None:
        style += [('FONT', (0,bold_row), (-1,bold_row), 'Helvetica-Bold', 9), ('BACKGROUND', (0,bold_row), (-1,bold_row), colors.HexColor('#e6f4ea'))]
    t.setStyle(TableStyle(style)); return t

def bullets(*items): return [Paragraph(x, B, bulletText='•') for x in items]

s = []
s.append(Paragraph('DeepOHeat-v1 sul Mac', TITLE))
s.append(Paragraph('Diario degli esperimenti: cosa abbiamo fatto girare, cosa è andato storto, cosa abbiamo scoperto. 28–29 agosto 2026.', SUB))

s.append(Paragraph('In due parole', H1))
s.append(Paragraph(
    'DeepOHeat-v1 è una rete neurale che, data la <b>mappa di potenza</b> di un chip (dove e quanto scaldano i componenti, '
    'una griglia 21×21), prevede la <b>temperatura in tutto il volume</b> del chip. È il surrogato veloce di un solutore numerico: '
    'millisecondi invece di secondi. Viene dal paper <i>DeepOHeat-v1: Efficient Operator Learning for Fast and Trustworthy Thermal '
    'Simulation and Optimization in 3D-IC Design</i> (Yu et al., UC Santa Barbara, IEEE TCPMT 2025, arXiv 2504.03955).', P))
s.append(Paragraph(
    'Due particolarità che spiegano quasi tutto quello che segue. <b>Uno</b>: la rete non impara da esempi "mappa → temperatura giusta", '
    'ma controllando che la temperatura che propone rispetti l\'equazione del calore (approccio <i>physics-informed</i>). '
    '<b>Due</b>: per essere veloce ricostruisce il campo 3D come somma di prodotti di funzioni 1D (struttura "separabile"), '
    'e questo è anche il motivo per cui su una CPU portatile un intero training dura circa un minuto.', P))

s.append(Paragraph('Setup e fix per farlo partire', H1))
s += bullets(
    'Ambiente: <font face="Courier">.venv</font> con Python 3.12, jax 0.11 (solo CPU), equinox, optax. Nessun <font face="Courier">requirements.txt</font> nella repo: l\'abbiamo scritto noi.',
    'La repo <b>non</b> contiene pesi pre-allenati né soluzioni di training: solo codice + dati di input (Google Drive). Tutti i modelli qui sotto sono allenati da zero.',
    'Fix 1 — <font face="Courier">jax.tree_map</font> rimosso nelle versioni recenti → <font face="Courier">jax.tree_util.tree_map</font>.',
    'Fix 2 — crash nell\'ottimizzatore: lo stato di Adam era inizializzato su tutti gli array del modello, i gradienti coprono solo quelli float. '
    'Le versioni nuove di optax non tollerano più la differenza → <font face="Courier">optimizer.init(eqx.filter(model, eqx.is_inexact_array))</font>.',
    'Aggiunti tre flag a <font face="Courier">heat_surface.py</font> (default = comportamento originale): '
    '<font face="Courier">--decay_steps</font>, <font face="Courier">--train_data</font>, <font face="Courier">--tag</font>.',
    'Tempo: ~7 ms per iterazione → 70 s per 10.000 epoche, 6 min per 50.000. Il paper riporta 30 s su RTX 3090.')

s.append(Paragraph('Esperimento 1 — Riproduzione del paper (default)', H1))
s.append(Paragraph(
    'I default della repo sono esattamente il setup del paper: 10.000 epoche, Adam lr 1e-3 con decadimento ×0,9 ogni 500 passi, 50 mappe per passo. '
    'Il test set sono 10 mappe di potenza "a blocchi" con soluzione di riferimento calcolata da un solutore commerciale (Celsius 3D).', P))
s.append(table([['', 'MAPE (metrica del paper)', 'rel. L2', 'tempo'],
                ['Paper (RTX 3090)', '0,049%', '—', '30 s'],
                ['Noi, default (CPU)', '0,058%', '2,96%', '70 s']], col_widths=[4.5*cm, 4.5*cm, 3*cm, 3*cm]))
s.append(Spacer(1, 6))
s.append(Paragraph(
    'Numeri allineati al paper. Ma guardando la <b>prima mappa di test</b> si vede il problema che il MAPE nasconde: la predizione azzecca il gradiente '
    'generale ma <b>liscia via i punti caldi</b>. Il picco vero è 33,5 °C, predetto 31,5 °C. Il MAPE è calcolato in Kelvin (300 K di base) e mediato '
    'sull\'intero volume: un errore locale di 2 °C ci sparisce dentro.', P))
s.append(fig(f'{R}_ep10000/sample0_true_vs_pred.png',
             caption='Fig. 1 — Mappa di test 0. Sopra: potenza in ingresso, temperatura vera sulla superficie superiore, predizione, errore. '
                     'Sotto: sezione verticale a metà chip. La mappa d\'errore ricalca la mappa di potenza: blu (sottostima) sotto i blocchi caldi.'))

s.append(Paragraph('Esperimento 2 — Più epoche (50.000)', H1))
s.append(Paragraph(
    'Prima idea, la più ovvia: allenare di più. Migliora, ma poco, e il log della loss spiega perché: da 20.000 epoche in avanti è piatta. '
    'Lo scheduler porta il learning rate a ~5e-6 già a 25.000 passi — il training si spegne da solo.', P))
s.append(table([['finestra di epoche', 'loss media', 'learning rate a fine finestra'],
                ['5k – 10k', '0,167', '1,2e-4'], ['10k – 15k', '0,096', '4,2e-5'], ['15k – 20k', '0,079', '1,5e-5'],
                ['20k – 25k', '0,068', '5,2e-6'], ['25k – 50k', '0,067 – 0,069', '< 2e-6']], col_widths=[4.5*cm, 4*cm, 5.5*cm]))
s.append(Spacer(1, 4))
s.append(Paragraph('Morale: con lo scheduler originale, 20–25k epoche è il punto giusto; oltre è tempo sprecato.', P))

s.append(Paragraph('Esperimento 3 — Decadimento del learning rate più lento', H1))
s.append(Paragraph(
    'Decadimento ogni 1.000 passi invece di 500, 50.000 epoche. Il learning rate resta utile per tutto il budget. '
    'La loss finale scende di 4× (0,067 → 0,015) e sulla prima mappa il picco è finalmente centrato (errore +0,1 °C).', P))
s.append(table([['', '10k, decay 500', '50k, decay 500', '50k, decay 1000'],
                ['rel. L2 medio test', '2,96%', '2,04%', '1,96%'],
                ['errore sul picco (max_l1)', '0,057', '0,027', '0,019'],
                ['mappa 0: errore sul picco', '−1,96 °C', '−0,71 °C', '+0,12 °C'],
                ['mappa 0: max |errore|', '2,39 °C', '2,03 °C', '1,24 °C']], col_widths=[5*cm, 3.2*cm, 3.2*cm, 3.4*cm]))
s.append(Spacer(1, 6))
s.append(fig(f'{R}_ep50000_d1000/sample0_runs_comparison.png',
             caption='Fig. 2 — Mappa di test 0, superficie superiore. Con il decadimento più lento (ultima colonna) i blocchi diventano riconoscibili.'))
s.append(Paragraph(
    'Un dato onesto però: la loss di training è scesa 4×, ma l\'errore <b>medio</b> sul test solo da 2,04% a 1,96%. Quindi l\'ottimizzazione non è più '
    'il collo di bottiglia. Il residuo è altrove.', P))

s.append(Paragraph('Intermezzo — Perché le predizioni sono sfocate?', H1))
s.append(Paragraph('Due cause, una fisica e una di dati.', P))
s.append(Paragraph('<b>La fisica smussa (un po\').</b> La mappa di potenza ha bordi a gradino, la temperatura no: il calore diffonde. '
                   'In superficie, dove entra il calore, la transizione al bordo di un blocco si spalma su ~0,2 di dominio; scendendo nello spessore i dettagli si fondono '
                   'fino a un fondo quasi uniforme. Quindi <i>l\'obiettivo non sono quadrati netti</i>: è la quantità giusta di smussamento.', P))
s.append(fig('results/results_surface/edge_profile.png',
             caption='Fig. 3 — Sinistra: la potenza lungo una linea che attraversa un blocco (gradino). Destra: la temperatura vera lungo la stessa linea a varie profondità. '
                     'Il "ginocchio" sul bordo è visibile solo in superficie.'))
s.append(Paragraph('<b>I dati di training sono di un altro tipo.</b> Il paper allena su <i>campi gaussiani casuali</i>: nuvole sfumate, con valori positivi e negativi. '
                   'Le mappe di test sono <i>rettangoli a gradino</i>. La rete non ha mai visto un bordo netto in allenamento: quando ne incontra uno, lo codifica con il '
                   'vocabolario delle sfumature. Il paper lo ammette a pag. 9: le mappe di test "differiscono significativamente" da quelle di training.', P))
s.append(fig('results/results_surface/train_vs_test_maps.png',
             caption='Fig. 4 — Sopra: 4 delle 10.000 mappe di training (campi gaussiani). Sotto: 4 delle 10 mappe di test (blocchi).'))

s.append(Paragraph('Esperimento 4 — Training set misto: gaussiani + blocchi', H1))
s.append(Paragraph(
    'Abbiamo scritto un generatore di mappe a blocchi (<font face="Courier">gen_block_maps.py</font>: 1–6 rettangoli casuali, potenza 0,25–2, bordi frazionari come nei test) '
    'e allenato su 5.000 gaussiani + 5.000 blocchi. Stesso modello, stessi iperparametri dell\'Esperimento 3: cambia solo il dato.', P))
s.append(table([['', 'solo gaussiani (paper)', 'misto gaussiani + blocchi'],
                ['rel. L2 medio test', '1,96%', '1,14%  (−42%)'],
                ['MAPE', '0,043%', '0,028%'],
                ['errore sul picco (max_l1)', '0,019', '0,015'],
                ['mappe migliorate', '—', '8 su 10']], col_widths=[5*cm, 4.5*cm, 5*cm], bold_row=1))
s.append(Spacer(1, 6))
s.append(fig(f'{R}_mixed/grf_vs_mixed.png',
             caption='Fig. 5 — Tre mappe di test. Le bande d\'errore lungo i bordi dei rettangoli si dimezzano con il training misto. I guadagni grossi sono sulle mappe '
                     'a pochi blocchi netti: mappa 2 da 3,9% a 0,7%, mappa 5 da 4,2% a 0,9%.'))
s.append(Paragraph('Unica mappa che non migliora: la <b>numero 9</b>, che ha potenza massima 4 mentre il generatore arrivava a 2. Fuori distribuzione in ampiezza.', P))

s.append(Paragraph('Esperimenti 5 e 6 — Solo blocchi? E potenza fino a 4?', H1))
s.append(Paragraph(
    'Se i blocchi aiutano, perché non allenare <i>solo</i> su blocchi, con potenza estesa a 4? L\'abbiamo provato (Esp. 5). Poi, per separare i due cambiamenti, '
    'anche il misto con potenza fino a 4 (Esp. 6).', P))
s.append(table([['training set', 'rel. L2 medio', 'max_l1', '|err. picco| medio', 'mappa 9: err. picco'],
                ['solo gaussiani (paper)', '1,96%', '0,019', '0,48 °C', '−2,79 °C'],
                ['misto, potenza ≤ 2', '1,14%', '0,015', '0,37 °C', '−2,43 °C'],
                ['misto, potenza ≤ 4', '1,10%', '0,011', '0,27 °C', '−1,34 °C'],
                ['100% blocchi, potenza ≤ 4', '4,42%', '0,069', '1,72 °C', '−2,84 °C']],
               col_widths=[4.6*cm, 2.6*cm, 2*cm, 3.2*cm, 3.2*cm], bold_row=3))
s.append(Spacer(1, 6))
s.append(fig(f'{R}_mixed_p4/datasets_comparison.png',
             caption='Fig. 6 — Stesso modello, quattro training set. Ultima colonna: il modello "solo blocchi" inventa una grossa macchia calda al centro della mappa 5 '
                     'dove non c\'è nulla. Penultima: il misto con potenza ≤ 4 mostra per la prima volta tutte e cinque le zone calde della mappa 0.'))
s.append(Paragraph('<b>Sorpresa</b>: 100% blocchi è il peggiore di tutti, peggio anche del paper, su tutte le 10 mappe. Non sfoca: <i>allucina</i>. '
                   'Il confronto con l\'Esp. 6 scagiona la potenza 4 (che anzi aiuta sui picchi): il colpevole è l\'<b>assenza dei campi gaussiani</b>.', P))
s.append(Paragraph('La spiegazione più probabile: l\'equazione è lineare nella potenza, e i campi gaussiani — densi, con segni misti, senza struttura ripetuta — costringono la rete a '
                   'imparare l\'operatore vero. I blocchi da soli sono sparsi e "tutti uguali": la rete può memorizzare come appare la temperatura sotto un rettangolo senza '
                   'imparare la fisica generale, e poi sbaglia appena la configurazione esce dalle statistiche viste. La scelta dei gaussiani nel paper, che sembrava strana, ha una ragione solida.', P))

s.append(Paragraph('Cosa abbiamo imparato', H1))
s += bullets(
    'Il codice riproduce il paper (con due fix di compatibilità). Il default a 10.000 epoche interrompe il training mentre sta ancora imparando.',
    'Con lo scheduler originale il punto giusto è 20–25k epoche; con decadimento ogni 1.000 passi si sfruttano 50k.',
    'Il MAPE in Kelvin è una metrica generosa: nasconde errori di ~2 °C sui picchi, che sono la grandezza che conta.',
    'La sfocatura è in buona parte un problema di <b>dati</b>: aggiungere blocchi al training set taglia l\'errore del 42% sulle stesse mappe di test del paper.',
    'Ma i campi gaussiani <b>servono</b>: senza, il modello allucina. Misto batte entrambi i puri.',
    'Il miglior modello: misto gaussiani + blocchi con potenza ≤ 4, 50k epoche, decadimento ogni 1.000 passi. MAPE 0,031%, rel. L2 1,10%, errore medio sui picchi 0,27 °C (paper: 0,049%).')

s.append(Paragraph('Cosa NON dimostrano questi esperimenti', H2))
s += bullets(
    'Il generatore di blocchi è stato disegnato <i>guardando</i> le 10 mappe di test. È una dimostrazione di meccanismo, non un confronto alla pari con il paper, il cui test era genuinamente fuori distribuzione.',
    'Non abbiamo un test set indipendente di blocchi con soluzione di riferimento: servirebbe un solutore numerico (quello del notebook richiede CUDA).',
    'Un solo seed per configurazione: differenze sotto lo 0,1% di rel. L2 (es. i due misti) sono rumore.',
    'Tutto sul caso "surface power". <font face="Courier">heat_volumetric.py</font> ha ricevuto gli stessi fix ma non è stato eseguito (0,5 h su GPU nel paper).')

s.append(Paragraph('Appendice — Riprodurre', H1))
s.append(Paragraph('Tutti i comandi dalla cartella <font face="Courier">DeepOHeat-v1/</font>, con il venv attivo. Ogni run salva pesi, predizioni e metriche in <font face="Courier">results/</font>.', P))
s.append(Paragraph(
    'python -m venv .venv &amp;&amp; source .venv/bin/activate &amp;&amp; pip install -r requirements.txt<br/>'
    '# Esp. 1 — default (= paper)<br/>python heat_surface.py<br/>'
    '# Esp. 3 — decadimento lento<br/>python heat_surface.py --epochs 50000 --decay_steps 1000<br/>'
    '# Esp. 6 — miglior modello<br/>python gen_block_maps.py --pmax 4 --seed 2 --out data/fs_train_surface_mixed_p4.npy<br/>'
    'python heat_surface.py --epochs 50000 --decay_steps 1000 --train_data data/fs_train_surface_mixed_p4.npy --tag _mixed_p4', CODE))
s.append(Paragraph('File toccati: <font face="Courier">heat_surface.py</font> (fix + 3 flag), <font face="Courier">heat_volumetric.py</font> (fix), '
                   '<font face="Courier">gen_block_maps.py</font> (nuovo), <font face="Courier">requirements.txt</font> (nuovo), <font face="Courier">.gitignore</font> (+ .venv).', P))

def footer(canvas, doc):
    canvas.saveState(); canvas.setFont('Helvetica', 8); canvas.setFillColor(colors.HexColor('#888888'))
    canvas.drawRightString(A4[0] - 1.8*cm, 1.1*cm, f'DeepOHeat-v1 — diario esperimenti — pag. {doc.page}'); canvas.restoreState()

doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=1.8*cm, rightMargin=1.8*cm, topMargin=1.8*cm, bottomMargin=1.8*cm,
                        title='DeepOHeat-v1 sul Mac — diario esperimenti', author='Alessandro Pitasi')
doc.build(s, onFirstPage=footer, onLaterPages=footer)
print(OUT)
