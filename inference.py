"""Inferencia tabular del primer modelo. No contiene una GNN."""
import pandas as pd
from pipeline import features


def score_transaction(bundle, transaction):
    if 'sender_prior_out' in bundle['columns']:
        raise ValueError('Este endpoint inicial no dispone de estado histórico.')
    row = dict(transaction)
    for k in ['From Bank', 'To Bank', 'Account', 'Account.1', 'Payment Currency',
              'Receiving Currency', 'Payment Format', 'Timestamp']:
        if not isinstance(row.get(k), str) or not row[k].strip():
            raise ValueError(f'Campo inválido: {k}')
    for k in ['Amount Paid', 'Amount Received']:
        row[k] = float(row[k])
        if not 0 <= row[k] < float('inf'):
            raise ValueError(f'Importe inválido: {k}')
    row['time'] = pd.to_datetime(row['Timestamp'], errors='raise')
    row['sender'] = (row['From Bank'], row['Account'])
    row['receiver'] = (row['To Bank'], row['Account.1'])
    x = features(pd.DataFrame([row]), include_history=False)[bundle['columns']]
    score = float(bundle['pipeline'].predict_proba(x)[0, 1])
    return {'score': score, 'threshold': bundle['threshold'],
            'alert': score >= bundle['threshold'], 'model': 'logistic_regression'}
