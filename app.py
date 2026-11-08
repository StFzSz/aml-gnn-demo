"""Panel académico: consulta, replay y comparación de resultados."""
import json
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
from inference import score_transaction

ROOT = Path(__file__).resolve().parent
BASELINE = ROOT/'outputs/ibm_baseline'
GNN = ROOT/'outputs/gnn'
st.set_page_config(page_title='Transferencias sospechosas', page_icon='🔎', layout='wide')
st.title('Detección de transferencias sospechosas')
st.caption('Prototipo académico · Datos sintéticos IBM · Flujo histórico simulado')

@st.cache_resource
def load_baseline():
    return joblib.load(BASELINE/'baseline.joblib')

@st.cache_resource
def load_gnn():
    from gnn_inference import SnapshotScorer
    return SnapshotScorer(GNN)

bundle = load_baseline()
scorer = None
assets = ['gnn.pt', 'metrics.json', 'edge_transform.joblib', 'context.npz', 'account_index.joblib', 'context.json']
if all((GNN/f).is_file() for f in assets):
    try:
        scorer = load_gnn()
    except (ImportError, OSError, ValueError, RuntimeError) as e:
        st.error(f'No se pudo cargar la GNN: {e}')
else:
    st.info('Modo de referencia tabular. Para habilitar la GNN, añadir sus artefactos entrenados en outputs/gnn.')
model_choice = st.selectbox('Modelo de inferencia', ['GraphSAGE', 'Regresión logística'] if scorer else ['Regresión logística'])
if model_choice == 'GraphSAGE':
    st.caption(f"Historial fijo al {scorer.meta['as_of']}; consultas anteriores a {scorer.meta['valid_before']}. "
               'Las conexiones no se actualizan durante el día.')


def predict(row):
    if model_choice == 'GraphSAGE':
        return scorer.score(row)
    import time
    start = time.perf_counter()
    result = score_transaction(bundle, row)
    result['latency_ms'] = (time.perf_counter()-start)*1000
    return result


def show_graph(row):
    path = GNN/'demo_graph.json'
    if not path.is_file():
        st.caption('Exportar ejemplos y conexiones con prepare_demo.py para habilitar la vista local.')
        return
    @st.cache_data
    def read_links():
        return json.loads(path.read_text())
    source = (row['From Bank'], row['Account'])
    target = (row['To Bank'], row['Account.1'])
    links = [e for e in read_links() if source in [tuple(e['source']), tuple(e['target'])]
             or target in [tuple(e['source']), tuple(e['target'])]][:20]
    # IDs internos cortos; etiquetas escapadas como strings DOT.
    nodes = {source, target}
    for e in links:
        nodes.update([tuple(e['source']), tuple(e['target'])])
    ids = {k: f'n{i}' for i, k in enumerate(sorted(nodes))}
    lines = ['digraph G { rankdir=LR; node [shape=box];']
    for k, value in ids.items():
        label = json.dumps(f'Banco {k[0]} / {k[1]}')
        lines.append(f'{value} [label={label}];')
    for e in links:
        lines.append(f"{ids[tuple(e['source'])]} -> {ids[tuple(e['target'])]};")
    lines.append(f'{ids[source]} -> {ids[target]} [style=dashed,label="Consulta"];')
    lines.append('}')
    st.graphviz_chart('\n'.join(lines))
    st.caption('Vista parcial de hasta 20 conexiones históricas. La línea discontinua es la consulta. '
               'Esta vista no es una explicación causal del score ni el grafo completo usado para entrenar.')


consult, replay, evaluation = st.tabs(['Consultar', 'Reproducir eventos', 'Resultados'])
encoder = bundle['pipeline'].named_steps['preprocess'].named_transformers_['categorical']
receiving, payment, formats = [list(v) for v in encoder.categories_]
with consult:
    with st.form('transfer'):
        left, right = st.columns(2)
        with left:
            paid = st.number_input('Importe enviado', min_value=0., value=500.)
            pc = st.selectbox('Moneda de envío', payment)
            sb = st.text_input('Banco emisor', '1')
            sa = st.text_input('Cuenta emisora', 'A')
            date = st.date_input('Fecha', value=pd.Timestamp('2022-09-09').date())
        with right:
            received = st.number_input('Importe recibido', min_value=0., value=500.)
            rc = st.selectbox('Moneda de recepción', receiving)
            tb = st.text_input('Banco receptor', '2')
            ta = st.text_input('Cuenta receptora', 'B')
            hour = st.time_input('Hora', value=pd.Timestamp('2022-09-09 12:00').time())
        fmt = st.selectbox('Forma de pago', formats)
        submitted = st.form_submit_button('Evaluar transferencia')
    if submitted:
        row = {'Timestamp': f'{date} {hour}', 'From Bank': sb, 'Account': sa,
               'To Bank': tb, 'Account.1': ta, 'Amount Paid': paid, 'Amount Received': received,
               'Payment Currency': pc, 'Receiving Currency': rc, 'Payment Format': fmt}
        try:
            result = predict(row)
            c1, c2, c3 = st.columns(3)
            c1.metric('Puntuación', f"{result['score']:.4f}")
            c2.metric('Umbral', f"{result['threshold']:.4f}")
            c3.metric('Inferencia local (ms)', f"{result['latency_ms']:.2f}")
            if result['alert']:
                st.warning('Alerta para revisión en esta demostración.')
            else:
                st.success('Sin alerta con el umbral seleccionado.')
            if result.get('accounts_outside_index'):
                st.info('Hay cuentas fuera del índice: se usa una representación de cuenta sin historial.')
            show_graph(row)
        except (ValueError, TypeError, KeyError) as e:
            st.error(str(e))
    st.caption('Puntuación sin calibrar; no es una probabilidad de delito. La latencia excluye carga del modelo, '
               'red y dibujo del grafo. La referencia tabular no utiliza conexiones.')

with replay:
    demo_path = GNN/'demo_transactions.json'
    if not demo_path.is_file():
        st.write('Faltan los ejemplos. Ejecutar prepare_demo.py después del entrenamiento.')
    else:
        @st.cache_data
        def demo_rows():
            return json.loads(demo_path.read_text())
        rows = demo_rows()
        st.write(f'{len(rows):,} eventos muestreados sin usar etiquetas, ordenados por fecha. '
                 'La evaluación global se muestra en Resultados.')
        state_key = f'replay_{model_choice}'
        if state_key not in st.session_state:
            st.session_state[state_key] = {'cursor': 0, 'results': []}
        state = st.session_state[state_key]
        a, b = st.columns(2)
        if a.button('Procesar 25 eventos'):
            for i in range(state['cursor'], min(state['cursor']+25, len(rows))):
                row = rows[i]
                r = predict(row)
                state['results'].append({'evento': i, 'fecha': row['Timestamp'],
                    'score': r['score'], 'alerta': r['alert'],
                    'etiqueta_dataset': int(row['Is Laundering']), 'latencia_ms': r['latency_ms']})
            state['cursor'] = len(state['results'])
        if b.button('Reiniciar reproducción'):
            state['cursor'], state['results'] = 0, []
        st.progress(state['cursor']/max(len(rows), 1))
        if state['results']:
            df = pd.DataFrame(state['results'])
            st.dataframe(df.tail(100), width='stretch')
            st.write(f"Alertas: {int(df.alerta.sum())} · Latencia p50: {df.latencia_ms.median():.2f} ms · "
                     f"p95: {df.latencia_ms.quantile(.95):.2f} ms (inferencia local).")
            index = st.selectbox('Inspeccionar conexiones del evento', df.evento.tolist())
            show_graph(rows[index])
        st.caption('Las etiquetas se muestran después de puntuar; nunca son entradas del modelo. '
                   'Se conserva el snapshot al inicio del día y no se simula actualización del historial.')

with evaluation:
    metrics = {}
    for name, folder in [('Regresión logística', BASELINE), ('GraphSAGE', GNN)]:
        if (folder/'metrics.json').is_file():
            metrics[name] = json.loads((folder/'metrics.json').read_text())
    st.write('Entrenamiento: 1–7 septiembre de 2022. Validación: 8. Prueba: 9–10. '
             'Umbrales elegidos en validación para cada modelo.')
    comparison = []
    for name, m in metrics.items():
        t = m['test']
        tn, fp, fn, tp = t['confusion_matrix_tn_fp_fn_tp']
        comparison.append({'modelo': name, 'AP': t['average_precision'], 'precisión': t['precision'],
            'recall': t['recall'], 'F1': t['f1'], 'detectadas': tp, 'falsas_alertas': fp, 'no_detectadas': fn})
    st.dataframe(pd.DataFrame(comparison), width='stretch', hide_index=True)
    st.caption('Menos falsas alertas puede implicar menos detecciones. Comparar también los umbrales y AP. '
               'Los resultados de la GNN disponibles en disco no implican que sus pesos estén cargados.')
    log = GNN/'training_log.json'
    if log.is_file():
        st.line_chart(pd.DataFrame(json.loads(log.read_text())).set_index('epoch')[['validation_average_precision']])
