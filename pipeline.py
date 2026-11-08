"""Primera etapa AML: auditoría y referencia temporal. No implementa aún la GNN."""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             precision_score, recall_score, f1_score, confusion_matrix)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

REQUIRED = ['Timestamp', 'From Bank', 'Account', 'To Bank', 'Account.1',
            'Amount Received', 'Receiving Currency', 'Amount Paid',
            'Payment Currency', 'Payment Format', 'Is Laundering']
BASE = ['log_paid', 'log_received', 'hour', 'weekday', 'same_bank',
        'same_account', 'Receiving Currency', 'Payment Currency', 'Payment Format']
HISTORY = ['sender_prior_out', 'sender_prior_in', 'receiver_prior_out',
           'receiver_prior_in', 'pair_prior_count']
CATEGORICAL = ['Receiving Currency', 'Payment Currency', 'Payment Format']


def load_csv(path, before=None):
    # Lectura como texto para preservar ceros y claves de cuenta/banco.
    df = pd.read_csv(path, dtype=str)
    missing = sorted(set(REQUIRED) - set(df.columns))
    if missing:
        raise ValueError(f'Columnas faltantes: {missing}')
    if df[REQUIRED].isna().any().any():
        raise ValueError('Hay valores faltantes: revisar antes de imputar.')
    df['time'] = pd.to_datetime(df['Timestamp'], format='%Y/%m/%d %H:%M', errors='raise')
    if before is not None:
        df = df.loc[df.time < pd.Timestamp(before)].copy()
    for col in ['Amount Paid', 'Amount Received']:
        df[col] = pd.to_numeric(df[col], errors='raise')
        if not np.isfinite(df[col]).all() or (df[col] < 0).any():
            raise ValueError(f'Importe inválido en {col}')
    df['label'] = pd.to_numeric(df['Is Laundering'], errors='raise')
    if not df['label'].isin([0, 1]).all():
        raise ValueError('Las etiquetas deben ser 0 o 1.')
    df['sender'] = list(zip(df['From Bank'], df['Account']))
    df['receiver'] = list(zip(df['To Bank'], df['Account.1']))
    return df.sort_values('time', kind='stable').reset_index(drop=True)


def split_temporal(df, train_end, val_end):
    a, b = pd.Timestamp(train_end), pd.Timestamp(val_end)
    if a >= b:
        raise ValueError('train_end debe ser anterior a val_end.')
    masks = [df.time < a, (df.time >= a) & (df.time < b), df.time >= b]
    for name, mask in zip(['train', 'validation', 'test'], masks):
        if not mask.any() or df.loc[mask, 'label'].nunique() != 2:
            raise ValueError(f'{name}: se necesitan observaciones de ambas clases; revisar cortes.')
    return masks


def features(df, include_history=True):
    x = pd.DataFrame(index=df.index)
    x['log_paid'] = np.log1p(df['Amount Paid'])
    x['log_received'] = np.log1p(df['Amount Received'])
    x['hour'] = df.time.dt.hour
    x['weekday'] = df.time.dt.dayofweek
    x['same_bank'] = (df['From Bank'] == df['To Bank']).astype(int)
    x['same_account'] = (df.sender == df.receiver).astype(int)
    x[CATEGORICAL] = df[CATEGORICAL]
    if not include_history:
        return x
    # Historial estrictamente anterior. Empates no se observan entre sí.
    # Actualizamos también con eventos de validación/test tras puntuarlos:
    # evaluación secuencial con pesos congelados, sin consultar sus etiquetas.
    outgoing, incoming, pairs = defaultdict(int), defaultdict(int), defaultdict(int)
    history = np.zeros((len(df), len(HISTORY)), dtype=np.int64)
    for _, group in df.groupby('time', sort=True):
        for i, s, r in zip(group.index, group.sender, group.receiver):
            history[i] = [outgoing[s], incoming[s], outgoing[r], incoming[r], pairs[s, r]]
        for s, r in zip(group.sender, group.receiver):
            outgoing[s] += 1
            incoming[r] += 1
            pairs[s, r] += 1
    x[HISTORY] = history
    return x


def report(df):
    accounts = set(df.sender) | set(df.receiver)
    return {'rows': len(df), 'accounts': len(accounts),
            'start': str(df.time.min()), 'end': str(df.time.max()),
            'positive_count': int(df.label.sum()), 'positive_rate': float(df.label.mean()),
            'duplicate_rows': int(df[REQUIRED].duplicated().sum()),
            'by_day': [{'day': str(day), 'rows': len(g), 'positives': int(g.label.sum())}
                       for day, g in df.groupby(df.time.dt.date)]}


def evaluate(y, p, threshold):
    pred = p >= threshold
    return {'average_precision': float(average_precision_score(y, p)),
            'precision': float(precision_score(y, pred, zero_division=0)),
            'recall': float(recall_score(y, pred, zero_division=0)),
            'f1': float(f1_score(y, pred, zero_division=0)),
            'confusion_matrix_tn_fp_fn_tp': confusion_matrix(y, pred, labels=[0, 1]).ravel().tolist(),
            'positive_rate': float(y.mean()), 'rows': len(y)}


def train(df, train_end, val_end, output, history=False):
    masks = split_temporal(df, train_end, val_end)
    cols = BASE + (HISTORY if history else [])
    x, y = features(df, include_history=history)[cols], df.label
    numeric = [c for c in cols if c not in CATEGORICAL]
    model = Pipeline([
        ('preprocess', ColumnTransformer([
            ('numeric', StandardScaler(), numeric),
            ('categorical', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL)])),
        ('classifier', LogisticRegression(class_weight='balanced', max_iter=2000, random_state=42))])
    model.fit(x.loc[masks[0]], y.loc[masks[0]])
    pv = model.predict_proba(x.loc[masks[1]])[:, 1]
    precision, recall, thresholds = precision_recall_curve(y.loc[masks[1]], pv)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    threshold = float(thresholds[int(np.argmax(f1))])
    results = {'model': 'logistic_regression', 'history_features': history,
               'threshold_selected_on_validation_f1': threshold,
               'train_end_exclusive': train_end, 'validation_end_exclusive': val_end,
               'data_start': str(df.time.min()), 'data_end': str(df.time.max()),
               'train_rows': int(masks[0].sum()), 'train_positives': int(y.loc[masks[0]].sum()),
               'validation': evaluate(y.loc[masks[1]], pv, threshold),
               'test': evaluate(y.loc[masks[2]], model.predict_proba(x.loc[masks[2]])[:, 1], threshold)}
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump({'pipeline': model, 'threshold': threshold, 'columns': cols}, output / 'baseline.joblib')
    (output / 'metrics.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['inspect', 'baseline'])
    parser.add_argument('--csv', type=Path, required=True)
    parser.add_argument('--train-end')
    parser.add_argument('--val-end')
    parser.add_argument('--history', action='store_true')
    parser.add_argument('--before', help='Excluir fecha y eventos posteriores antes de entrenar, por ejemplo 2022-09-11.')
    parser.add_argument('--output', type=Path, default=Path('outputs/initial'))
    args = parser.parse_args()
    df = load_csv(args.csv, before=args.before)
    if args.command == 'inspect':
        result = report(df)
    else:
        if not args.train_end or not args.val_end:
            parser.error('baseline requiere --train-end y --val-end; elegir tras inspect.')
        result = train(df, args.train_end, args.val_end, args.output, args.history)
    # Hash por bloques: no carga otra copia del archivo en RAM.
    digest = hashlib.sha256()
    with args.csv.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    result['input_sha256'] = digest.hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'run.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
