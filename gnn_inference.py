"""Inferencia CPU de GNN con un snapshot fijo, sin recalcular el grafo."""
import json
import hashlib
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


class CompactAccountIndex:
    def __init__(self, folder, parts):
        self.keys = np.concatenate([np.load(folder/p['keys'], allow_pickle=False) for p in parts])
        self.ids = np.concatenate([np.load(folder/p['ids'], allow_pickle=False) for p in parts])
        if len(self.keys)!=len(self.ids) or not np.array_equal(self.keys,np.sort(self.keys)) or np.any(self.keys[1:]==self.keys[:-1]):
            raise ValueError('Índice compacto inválido.')

    def __len__(self):
        return len(self.keys)

    def get(self, key, default=-1):
        encoded=json.dumps(list(key),ensure_ascii=False,separators=(',',':')).encode('utf-8')
        value=np.array(hashlib.sha256(encoded).digest()[:16],dtype='V16')
        i=int(np.searchsorted(self.keys,value))
        return int(self.ids[i]) if i<len(self.keys) and self.keys[i]==value else default


class ShardedEmbeddings:
    def __init__(self,folder,parts):
        self.parts=[np.load(folder/p['file'],mmap_mode='r',allow_pickle=False) for p in parts]
        self.starts=np.array([p['start'] for p in parts],dtype=np.int64)
        self.rows=sum(len(p) for p in self.parts)
        if not self.parts or self.starts[0]!=0 or any(
            self.starts[i]!=self.starts[i-1]+len(self.parts[i-1]) for i in range(1,len(self.parts))):
            raise ValueError('Partes de representaciones incompatibles.')
        self.hidden=self.parts[0].shape[1]

    def __len__(self):
        return self.rows

    def __getitem__(self,ids):
        ids=np.asarray(ids,dtype=np.int64)
        if (ids<0).any() or (ids>=self.rows).any():
            raise IndexError('Cuenta fuera de representaciones.')
        output=np.empty((len(ids),self.hidden),dtype=np.float32)
        part_ids=np.searchsorted(self.starts,ids,side='right')-1
        for i in np.unique(part_ids):
            mask=part_ids==i
            output[mask]=self.parts[i][ids[mask]-self.starts[i]]
        return output


class SnapshotScorer:
    def __init__(self, folder):
        folder = Path(folder)
        self.meta = json.loads((folder/'context.json').read_text(encoding='utf-8-sig'))
        sharded = self.meta.get('storage') == 'sharded_float16_v1'
        assets = ['gnn.pt','metrics.json','edge_transform.joblib','context.json'] if sharded else REQUIRED_ASSETS
        for filename in assets:
            if not (folder / filename).is_file():
                raise FileNotFoundError(f'Falta outputs/gnn/{filename}')
        checkpoint = torch.load(folder/'gnn.pt', map_location='cpu', weights_only=True)
        self.model = AMLGraphSAGE(checkpoint['edge_features'], checkpoint['hidden'])
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()
        self.transform = joblib.load(folder/'edge_transform.joblib')
        if sharded:
            self.account_index = CompactAccountIndex(folder,self.meta['account_parts'])
            self.embeddings = ShardedEmbeddings(folder,self.meta['embeddings_parts'])
        else:
            self.account_index = {tuple(k): i for i, k in enumerate(joblib.load(folder/'account_index.joblib'))}
            with np.load(folder/'context.npz') as context:
                self.embeddings = context['embeddings'].copy()
        if len(self.embeddings) != len(self.account_index):
            raise ValueError('Índice de cuentas y representaciones incompatibles.')
        self.threshold = float(json.loads((folder/'metrics.json').read_text(encoding='utf-8-sig'))['threshold'])
        # Cuenta sin historial: features (0,0,0,1), sin vecinos en ninguna capa.
        with torch.inference_mode():
            self.unknown = self.model.encode(torch.tensor([[0., 0., 0., 1.]]),
                                             torch.empty((2, 0), dtype=torch.long))[0].numpy()

    def score_batch(self, transactions, batch_size=4096):
        """Puntuar un lote del periodo del snapshot; no usa etiquetas."""
        start = time.perf_counter()
        df = pd.DataFrame(transactions).copy().reset_index(drop=True)
        required = ['From Bank', 'To Bank', 'Account', 'Account.1', 'Payment Currency',
                    'Receiving Currency', 'Payment Format', 'Timestamp', 'Amount Paid', 'Amount Received']
        if df.empty:
            raise ValueError('El lote está vacío.')
        if set(required)-set(df.columns):
            raise ValueError(f'Columnas faltantes: {sorted(set(required)-set(df.columns))}')
        if df[required].isna().any().any():
            raise ValueError('Hay valores faltantes en el lote.')
        for key in required[:8]:
            if not df[key].map(lambda v: isinstance(v, str) and bool(v.strip())).all():
                raise ValueError(f'Campo inválido: {key}. Conservar identificadores como texto.')
        for key in ['Amount Paid', 'Amount Received']:
            df[key] = pd.to_numeric(df[key], errors='raise')
            if not np.isfinite(df[key]).all() or (df[key]<0).any():
                raise ValueError(f'Importe inválido: {key}')
        df['time'] = pd.to_datetime(df.Timestamp, format='mixed', errors='raise')
        if df.time.isna().any() or df.time.dt.tz is not None:
            raise ValueError('Fechas inválidas o con zona horaria; usar la hora del dataset.')
        valid = ((df.time >= pd.Timestamp(self.meta['as_of'])) &
                 (df.time < pd.Timestamp(self.meta['valid_before'])))
        if not valid.all():
            raise ValueError(f"{int((~valid).sum())} eventos fuera del snapshot: "
                             f"{self.meta['as_of']} a {self.meta['valid_before']} (exclusivo).")
        df['sender'] = list(zip(df['From Bank'], df.Account))
        df['receiver'] = list(zip(df['To Bank'], df['Account.1']))
        raw = features(df, include_history=False)[BASE]
        if self.meta.get('storage') == 'sharded_float16_v1':
            numeric = [c for c in BASE if c not in ['Receiving Currency','Payment Currency','Payment Format']]
            raw[numeric] = raw[numeric].astype(np.float32)
        predictions, missing_counts = [], []
        for start_idx in range(0,len(df),batch_size):
            part = df.iloc[start_idx:start_idx+batch_size]
            attrs = torch.as_tensor(self.transform.transform(raw.iloc[start_idx:start_idx+batch_size]).astype(np.float32))
            vectors, misses = [], np.zeros(len(part),dtype=np.int32)
            for column in ['sender', 'receiver']:
                ids = np.array([self.account_index.get(k,-1) for k in part[column]],dtype=np.int64)
                unknown = ids < 0
                z = self.embeddings[np.maximum(ids,0)].copy()
                z[unknown] = self.unknown
                vectors.append(torch.from_numpy(z))
                misses += unknown
            with torch.inference_mode():
                logits = self.model.head(torch.cat([vectors[0],vectors[1],attrs],dim=1)).flatten()
                predictions.append(torch.sigmoid(logits).numpy())
            missing_counts.append(misses)
        scores = np.concatenate(predictions)
        result = pd.DataFrame({'score': scores, 'alert': scores >= self.threshold,
                               'accounts_outside_index': np.concatenate(missing_counts)})
        result.attrs['elapsed_ms'] = (time.perf_counter()-start)*1000
        result.attrs['rows'] = len(result)
        return result

    def score(self, transaction):
        batch = self.score_batch([transaction])
        row = batch.iloc[0]
        return {'model': 'GraphSAGE_edge_classifier', 'score': float(row.score),
                'threshold': self.threshold, 'alert': bool(row.alert),
                'latency_ms': batch.attrs['elapsed_ms'],
                'accounts_outside_index': int(row.accounts_outside_index),
                'context_as_of': self.meta['as_of']}
