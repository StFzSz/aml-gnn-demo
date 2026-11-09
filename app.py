"""Panel AML: demostración, consulta, resultados y procesamiento por lotes."""
import json
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from inference import score_transaction

ROOT = Path(__file__).resolve().parent
BASELINE, GNN = ROOT/'outputs/ibm_baseline', ROOT/'outputs/gnn'
st.set_page_config(page_title='Monitor de transferencias', page_icon='🔎', layout='wide')
st.title('Monitor de transferencias')
st.caption('Detecta operaciones para revisar y explora sus conexiones · Demostración con datos sintéticos IBM')

@st.cache_resource
def baseline_model():
    return joblib.load(BASELINE/'baseline.joblib')

@st.cache_resource
def gnn_model():
    from gnn_inference import SnapshotScorer
    return SnapshotScorer(GNN)

@st.cache_data
def read_json(path):
    return json.loads(Path(path).read_text())

bundle = baseline_model()
scorer = None
assets = ['gnn.pt','metrics.json','edge_transform.joblib','context.npz','account_index.joblib','context.json']
if all((GNN/name).is_file() for name in assets):
    try:
        scorer = gnn_model()
    except (ImportError, ValueError, OSError, RuntimeError) as e:
        st.error(f'No se pudo cargar GraphSAGE: {e}')

with st.sidebar:
    st.subheader('Configuración')
    model = st.selectbox('Modelo', ['GraphSAGE','Regresión logística'] if scorer else ['Regresión logística'])
    st.caption('GraphSAGE incorpora el historial de conexiones. La regresión es la referencia tabular.')
    official = scorer.threshold if model == 'GraphSAGE' else float(bundle['threshold'])
    explore_threshold = st.checkbox('Explorar otro umbral', value=False,
        help='Solo modifica las alertas de la demostración; las métricas oficiales permanecen iguales.')
    slider_value = st.slider('Umbral para marcar una alerta',0.,1.,float(official),.001,disabled=not explore_threshold)
    threshold = slider_value if explore_threshold else official
    if explore_threshold:
        st.warning('Umbral exploratorio. No modifica el modelo ni sus resultados oficiales.')
    st.caption('Un umbral menor suele producir más alertas; uno mayor, menos.')
    st.divider()
    st.write('Los datos incluyen más de 5 millones de operaciones. La muestra web es más pequeña para facilitar la demostración.')
    if model == 'GraphSAGE':
        st.caption(f"Historial fijo al {scorer.meta['as_of']}. Solo admite consultas anteriores a {scorer.meta['valid_before']}.")

view = st.radio('Vista', ['Demostración','Consultar una operación','Comparar modelos','Procesar un CSV'],horizontal=True)


def predict_many(rows):
    started = time.perf_counter()
    if model == 'GraphSAGE':
        scores = scorer.score_batch(rows)['score'].to_numpy(dtype=float)
    else:
        scores = np.array([score_transaction(bundle,row)['score'] for row in rows])
    return scores, (time.perf_counter()-started)*1000


def show_connections(row):
    path = GNN/'demo_graph.json'
    if not path.is_file():
        st.info('No se exportaron conexiones para esta demostración.')
        return
    source = (row['From Bank'],row['Account'])
    target = (row['To Bank'],row['Account.1'])
    links = [e for e in read_json(str(path)) if source in [tuple(e['source']),tuple(e['target'])]
             or target in [tuple(e['source']),tuple(e['target'])]][:20]
    nodes = {source,target}
    for edge in links:
        nodes.update([tuple(edge['source']),tuple(edge['target'])])
    aliases = {key:f'C{i+1}' for i,key in enumerate(sorted(nodes))}
    lines = ['digraph G { rankdir=LR; bgcolor="transparent"; node [shape=box,style="rounded,filled",fillcolor="#e8f1ff",fontcolor="#14233b"]; edge [color="#7890b0"];']
    for key, label in aliases.items():
        role = 'Emisor' if key == source else ('Receptor' if key == target else 'Conexión')
        if source == target == key:
            role = 'Emisor y receptor'
        lines.append(f'{label} [label={json.dumps(label+": "+role)}];')
    for edge in links:
        lines.append(f"{aliases[tuple(edge['source'])]} -> {aliases[tuple(edge['target'])]};")
    lines.append(f'{aliases[source]} -> {aliases[target]} [style=dashed,color="#d97a12",label="Operación consultada",fontcolor="#d97a12"];')
    lines.append('}')
    st.graphviz_chart('\n'.join(lines))
    st.caption('Flechas continuas: algunas transferencias del historial. Línea discontinua: operación consultada. '
               'Se muestran como máximo 20 conexiones; no es una explicación causal del score.')
    with st.expander('Ver las cuentas detrás de C1, C2…'):
        st.dataframe(pd.DataFrame([{'Referencia':v,'Banco':k[0],'Cuenta':k[1]} for k,v in aliases.items()]),hide_index=True)


def readable_table(rows, scores):
    raw = pd.DataFrame(rows)
    table = pd.DataFrame({'Fecha':raw.Timestamp,
        'Emisor':raw['From Bank'].astype(str)+' / '+raw.Account.astype(str),
        'Receptor':raw['To Bank'].astype(str)+' / '+raw['Account.1'].astype(str),
        'Importe enviado':pd.to_numeric(raw['Amount Paid']), 'Moneda':raw['Payment Currency'],
        'Puntuación':scores, 'Estado':np.where(scores>=threshold,'Revisar','Sin alerta')})
    return table


if view == 'Demostración':
    path = GNN/'demo_transactions.json'
    if not path.is_file():
        st.info('Ejecuta prepare_demo.py para añadir ejemplos.')
    else:
        rows = read_json(str(path))
        state = st.session_state.setdefault('demo_'+model, {'cursor':0,'scores':[], 'last_ms':None,'last_n':0})
        st.subheader('Revisa un flujo de transferencias')
        st.write(f'Muestra de {len(rows):,} operaciones del periodo de prueba. Cada lote se evalúa al pulsar el botón.')
        a,b,c = st.columns([2,2,1])
        size = a.selectbox('Operaciones por lote',[25,100,500,1000],index=1)
        advance = b.button('Procesar siguiente lote',type='primary')
        all_events = b.button('Procesar toda la muestra')
        reset = c.button('Reiniciar')
        if reset:
            state.update(cursor=0,scores=[],last_ms=None,last_n=0)
        if (advance or all_events) and state['cursor'] < len(rows):
            end = len(rows) if all_events else min(len(rows),state['cursor']+size)
            with st.spinner('Analizando transferencias…'):
                scores, elapsed = predict_many(rows[state['cursor']:end])
            state['scores'].extend(scores.tolist())
            state['last_ms'],state['last_n'],state['cursor'] = elapsed,len(scores),end
        n = state['cursor']
        scores = np.asarray(state['scores'])
        alerts = int((scores>=threshold).sum())
        c1,c2,c3,c4 = st.columns(4)
        c1.metric('Operaciones analizadas',f'{n:,}')
        c2.metric('Para revisar',f'{alerts:,}')
        c3.metric('Porcentaje con alerta',f'{100*alerts/n:.1f} %' if n else '—')
        c4.metric('Tiempo del último lote',f"{state['last_ms']:.1f} ms" if state['last_ms'] is not None else '—')
        st.progress(n/max(len(rows),1),text=f'{n:,} de {len(rows):,} operaciones')
        st.caption('El tiempo del lote incluye transformación e inferencia en el servidor. Excluye red, carga inicial y dibujo. '
                   'No equivale a la latencia individual ni a p50/p95.')
        if n:
            table = readable_table(rows[:n],scores)
            table.index.name = 'Evento'
            only = st.selectbox('Mostrar operaciones',['Todas','Solo alertas','Sin alerta'])
            shown = table if only == 'Todas' else table[table.Estado == ('Revisar' if only == 'Solo alertas' else 'Sin alerta')]
            st.dataframe(shown.tail(100),width='stretch',column_config={
                'Puntuación':st.column_config.NumberColumn(format='%.3f'),
                'Importe enviado':st.column_config.NumberColumn(format='%.2f')})
            st.download_button('Descargar operaciones analizadas',table.to_csv(index=True).encode(),
                               'operaciones_analizadas.csv','text/csv')
            if shown.empty:
                st.info('No hay operaciones que cumplan este filtro. Puedes procesar el resto de la muestra.')
            else:
                index = st.selectbox('Inspeccionar una operación',shown.index.tolist(),
                    format_func=lambda i:f"Evento {i} · {rows[i]['Timestamp']} · {table.loc[i,'Estado']}")
                st.write(f"**Puntuación:** {scores[index]:.3f} · **Umbral:** {threshold:.3f}")
                show_connections(rows[index])
            with st.expander('Mostrar etiquetas del dataset para comprobar resultados'):
                labels = np.array([int(row['Is Laundering']) for row in rows[:n]])
                st.write(f'Positivas en este tramo: {int(labels.sum())}. '
                         f'Detectadas con el umbral mostrado: {int(((labels==1)&(scores>=threshold)).sum())}.')
                st.caption('Las etiquetas no entran al modelo. Esta muestra no sustituye el test completo.')
        else:
            st.info('Pulsa «Procesar siguiente lote» para empezar.')
        with st.expander('¿Qué significa una alerta?'):
            st.write('Significa que la puntuación supera el umbral seleccionado. Es una señal para revisar, '
                     'no una confirmación de fraude. La puntuación no está calibrada como probabilidad. '
                     'El historial de la GNN permanece fijo durante el día de demostración.')

elif view == 'Consultar una operación':
    st.subheader('Introduce una transferencia')
    encoder = bundle['pipeline'].named_steps['preprocess'].named_transformers_['categorical']
    receiving,payment,formats = [list(v) for v in encoder.categories_]
    with st.form('query'):
        left,right = st.columns(2)
        with left:
            sb=st.text_input('Banco emisor','1'); sa=st.text_input('Cuenta emisora','A')
            paid=st.number_input('Importe enviado',min_value=0.,value=500.)
            pc=st.selectbox('Moneda de envío',payment)
            day=st.date_input('Fecha',value=pd.Timestamp('2022-09-09').date())
        with right:
            tb=st.text_input('Banco receptor','2'); ta=st.text_input('Cuenta receptora','B')
            received=st.number_input('Importe recibido',min_value=0.,value=500.)
            rc=st.selectbox('Moneda de recepción',receiving)
            hour=st.time_input('Hora',value=pd.Timestamp('2022-09-09 12:00').time())
        fmt=st.selectbox('Forma de pago',formats)
        submit=st.form_submit_button('Analizar operación',type='primary')
    if submit:
        row={'Timestamp':f'{day} {hour}','From Bank':sb,'Account':sa,'To Bank':tb,'Account.1':ta,
             'Amount Paid':paid,'Amount Received':received,'Payment Currency':pc,'Receiving Currency':rc,'Payment Format':fmt}
        try:
            scores,elapsed=predict_many([row]); value=float(scores[0])
            if value>=threshold:
                st.warning('Para revisar: supera el umbral seleccionado.')
            else:
                st.success('Sin alerta con el umbral seleccionado.')
            st.metric('Puntuación del modelo',f'{value:.3f}')
            st.caption(f'Umbral: {threshold:.3f} · Procesamiento: {elapsed:.2f} ms. '
                       'Las cuentas fuera del índice usan una representación sin historial.')
            show_connections(row)
        except (ValueError,KeyError,TypeError) as e:
            st.error(str(e))

elif view == 'Comparar modelos':
    st.subheader('Resultados sobre operaciones de prueba')
    st.write('Ambos modelos evaluaron los días 9–10 de septiembre de 2022. '
             'Se entrenaron con los días 1–7 y eligieron su umbral en el día 8.')
    comparison=[]
    for name,folder in [('Regresión logística',BASELINE),('GraphSAGE',GNN)]:
        path=folder/'metrics.json'
        if path.is_file():
            m=read_json(str(path)); t=m.get('test')
            if t:
                tn,fp,fn,tp=t['confusion_matrix_tn_fp_fn_tp']
                comparison.append({'Modelo':name,'Precisión de alertas':100*t['precision'],
                    'Detección (recall)':100*t['recall'],'F1':t['f1'],'Average precision':t['average_precision'],
                    'Positivas detectadas':tp,'Falsas alertas':fp,'Positivas no detectadas':fn})
    frame=pd.DataFrame(comparison)
    st.dataframe(frame,width='stretch',hide_index=True,column_config={
        'Precisión de alertas':st.column_config.NumberColumn(format='%.2f %%'),
        'Detección (recall)':st.column_config.NumberColumn(format='%.2f %%')})
    if len(frame):
        metric=st.selectbox('Comparar visualmente',['Detección (recall)','Precisión de alertas','Average precision','F1'])
        st.bar_chart(frame.set_index('Modelo')[[metric]])
    st.info('Menos falsas alertas puede significar que también se detectan menos casos. '
            'La GNN inicial no supera a la referencia en todas las métricas.')
    with st.expander('Entender las métricas',expanded=True):
        st.write('**Precisión de alertas:** de las operaciones marcadas, qué porcentaje tiene etiqueta positiva. '
                 '**Detección (recall):** de todos los positivos del test, qué porcentaje encontró el modelo. '
                 '**F1:** combina precisión y recall. **Average precision:** resume la calidad del ordenamiento '
                 'a distintos umbrales; un valor mayor es mejor.')
    log=GNN/'training_log.json'
    if log.is_file():
        with st.expander('Ver evolución del entrenamiento'):
            st.line_chart(pd.DataFrame(read_json(str(log))).set_index('epoch')[['validation_average_precision']])
    st.caption('Los umbrales exploratorios de la barra lateral no modifican estos resultados oficiales.')

elif view == 'Procesar un CSV':
    st.subheader('Analiza operaciones por lotes')
    st.write('Carga un CSV pequeño para consultar el modelo publicado. Para el archivo completo, '
             'usa el procesamiento local con un historial reconstruido para cada día.')
    if model == 'GraphSAGE':
        st.info(f"Este servicio usa un snapshot fijo: {scorer.meta['as_of']} a {scorer.meta['valid_before']} (exclusivo). "
                'Los eventos de otras fechas se rechazan; no se puntúan con un historial incorrecto.')
    upload=st.file_uploader('CSV de transferencias (máximo 20 MB)',type=['csv'])
    if upload is not None:
        if upload.size>20*1024*1024:
            st.error('El archivo supera 20 MB. Usa el procesamiento local para bases grandes.')
        elif st.button('Analizar CSV',type='primary'):
            try:
                frame=pd.read_csv(upload,dtype=str)
                if len(frame)>50000:
                    raise ValueError('El panel admite hasta 50.000 operaciones por archivo.')
                scores,elapsed=predict_many(frame.to_dict('records'))
                output=readable_table(frame.to_dict('records'),scores)
                st.success(f'{len(output):,} operaciones analizadas en {elapsed/1000:.2f} s.')
                st.metric('Alertas para revisar',int((scores>=threshold).sum()))
                st.dataframe(output.head(100),width='stretch')
                st.download_button('Descargar resultado CSV',output.to_csv(index=False).encode(),'resultado.csv','text/csv')
            except (ValueError,TypeError,KeyError) as e:
                st.error(str(e))
    demo=GNN/'demo_transactions.json'
    if demo.is_file():
        example=pd.DataFrame(read_json(str(demo))[:10]).drop(columns=['Is Laundering'],errors='ignore')
        st.download_button('Descargar ejemplo de entrada',example.to_csv(index=False).encode(),'ejemplo_transferencias.csv','text/csv')
    with st.expander('Procesar toda la base en tu computadora'):
        st.write('score_dataset.py recorre la ventana de datos y reconstruye un snapshot por día. '
                 'La opción test evalúa los días de prueba; all incluye también entrenamiento y validación '
                 'y los identifica por separado. Los datos posteriores a la ventana principal quedan excluidos y se cuentan.')
        st.code('.\\.venv\\Scripts\\python.exe score_dataset.py --csv .\\data\\raw\\HI-Small_Trans.csv --scope test --device cuda',language='powershell')
        st.caption('El CSV completo no se sube a Streamlit. Los pesos quedan congelados; este paso no reentrena.')
    summary=st.file_uploader('Ver resumen JSON de un procesamiento local',type=['json'],key='summary_upload')
    if summary is not None:
        try:
            result=json.load(summary)
            if result.get('schema') != 'aml_batch_v1':
                raise ValueError('El archivo no es un resumen de score_dataset.py.')
            st.write(f"Operaciones procesadas: {result['scored_rows']:,}. Alcance: {result['scope']}.")
            st.dataframe(pd.DataFrame(result['by_day']),width='stretch',hide_index=True)
        except (ValueError,KeyError,TypeError) as e:
            st.error(str(e))
