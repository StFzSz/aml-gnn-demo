"""Inferencia CPU de GNN con un snapshot fijo, sin recalcular el grafo."""
import json
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from gnn_model import AMLGraphSAGE
from pipeline import BASE, features

REQUIRED_ASSETS = ['gnn.pt', 'metrics.json', 'edge_transform.joblib', 'context.npz',
                   'account_index.joblib', 'context.json']


class SnapshotScorer:
    def __init__(self, folder):
        folder = Path(folder)
        for filename in REQUIRED_ASSETS:
            if not (folder / filename).is_file():
                raise FileNotFoundError(f'Falta outputs/gnn/{filename}')
        checkpoint = torch.load(folder/'gnn.pt', map_location='cpu', weights_only=True)
        self.model = AMLGraphSAGE(checkpoint['edge_features'], checkpoint['hidden'])
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()
        self.transform = joblib.load(folder/'edge_transform.joblib')
        self.account_index = {tuple(k): i for i, k in enumerate(joblib.load(folder/'account_index.joblib'))}
        with np.load(folder/'context.npz') as context:
            self.embeddings = context['embeddings'].copy()
        if len(self.embeddings) != len(self.account_index):
            raise ValueError('Índice de cuentas y representaciones incompatibles.')
        self.meta = json.loads((folder/'context.json').read_text())
        self.threshold = float(json.loads((folder/'metrics.json').read_text())['threshold'])
        # Cuenta sin historial: features (0,0,0,1), sin vecinos en ninguna capa.
        with torch.inference_mode():
            self.unknown = self.model.encode(torch.tensor([[0., 0., 0., 1.]]),
                                             torch.empty((2, 0), dtype=torch.long))[0].numpy()

    def score(self, transaction):
        start = time.perf_counter()
        row = dict(transaction)
        for key in ['From Bank', 'To Bank', 'Account', 'Account.1', 'Payment Currency',
                    'Receiving Currency', 'Payment Format', 'Timestamp']:
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f'Campo inválido: {key}')
        for key in ['Amount Paid', 'Amount Received']:
            row[key] = float(row[key])
            if not np.isfinite(row[key]) or row[key] < 0:
                raise ValueError(f'Importe inválido: {key}')
        row['time'] = pd.Timestamp(row['Timestamp'])
        if row['time'].tzinfo is not None:
            raise ValueError('Usar la hora del dataset sin zona horaria.')
        if not pd.Timestamp(self.meta['as_of']) <= row['time'] < pd.Timestamp(self.meta['valid_before']):
            raise ValueError(f"Snapshot válido desde {self.meta['as_of']} hasta {self.meta['valid_before']} (exclusivo).")
        row['sender'] = (row['From Bank'], row['Account'])
        row['receiver'] = (row['To Bank'], row['Account.1'])
        raw = features(pd.DataFrame([row]), include_history=False)[BASE]
        attrs = torch.as_tensor(self.transform.transform(raw).astype(np.float32))
        vectors = []
        missing = 0
        for key in [row['sender'], row['receiver']]:
            idx = self.account_index.get(key)
            vectors.append(self.unknown if idx is None else self.embeddings[idx])
            missing += int(idx is None)
        z = torch.from_numpy(np.stack(vectors))
        with torch.inference_mode():
            score = torch.sigmoid(self.model.score(z, torch.tensor([[0], [1]]), attrs)).item()
        return {'model': 'GraphSAGE_edge_classifier', 'score': score,
                'threshold': self.threshold, 'alert': score >= self.threshold,
                'latency_ms': (time.perf_counter() - start) * 1000,
                'accounts_outside_index': missing, 'context_as_of': self.meta['as_of']}
