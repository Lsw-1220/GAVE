"""Bounded-memory CSV conversion and disk-backed trajectory sampling."""
import ast
import hashlib
import json
import pickle
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
from .utils import EpisodeReplayBuffer, getScore


class DiskReplayBuffer(EpisodeReplayBuffer):
    def __init__(self, state_dim, act_dim, data_path, cache_dir, chunksize=10000, K=20, scale=2000):
        self.device, self.state_dim, self.act_dim = 'cpu', state_dim, act_dim
        self.K, self.scale = K, scale
        paths = [Path(p).resolve() for p in ([data_path] if isinstance(data_path, str) else data_path)]
        signature = [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in paths]
        key = hashlib.sha256(json.dumps([1, signature]).encode()).hexdigest()[:20]
        folder = Path(cache_dir)
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / (key + '.sqlite')
        if not target.exists():
            temporary = target.with_suffix('.building')
            temporary.unlink(missing_ok=True)
            db = sqlite3.connect(str(temporary))
            db.execute('CREATE TABLE episodes (id INTEGER PRIMARY KEY, length INTEGER, data BLOB)')
            db.execute('CREATE TABLE metadata (data BLOB)')
            count, total = 0, 0
            mean, m2 = np.zeros(state_dim), np.zeros(state_dim)
            columns = ['state', 'action', 'reward', 'done', 'next_state', 'budget', 'CPAConstraint']
            pending = []
            chunks = 0
            for path in paths:
                for chunk in pd.read_csv(path, usecols=columns, chunksize=chunksize):
                    for row in chunk[columns].itertuples(index=False, name=None):
                        pending.append(row)
                        if not row[3]:
                            if len(pending) > 10000:
                                raise ValueError(f'Episode exceeds 10000 rows in {path}; check done flags')
                            continue
                        if len(pending) > 1:
                            states = np.asarray([ast.literal_eval(r[0]) for r in pending], dtype=np.float64)
                            ns = [ast.literal_eval(r[4]) for r in pending[:-1]]
                            ns.append(ns[-1])  # Preserve original terminal next-state convention.
                            ns = np.asarray(ns, dtype=np.float64)
                            rewards = np.asarray([r[2] for r in pending], dtype=np.float64).reshape(-1, 1)
                            cumulative = np.r_[0., np.cumsum(rewards)]
                            scores = getScore(pending[0][5], pending[0][6], np.vstack([states, ns[-1]]), cumulative)
                            trajectory = dict(observations=states, next_states=ns,
                                actions=np.asarray([r[1] for r in pending]).reshape(-1, 1), rewards=rewards,
                                dones=np.asarray([r[3] for r in pending]), all_reward=cumulative,
                                curr_score=scores[-1]-scores)
                            n = len(states)
                            delta = states.mean(0) - mean
                            m2 += ((states-states.mean(0))**2).sum(0) + delta**2 * total*n/(total+n)
                            mean += delta*n/(total+n)
                            total += n
                            db.execute('INSERT INTO episodes VALUES (?, ?, ?)', (count, n, pickle.dumps(trajectory, protocol=4)))
                            count += 1
                        pending = []
                    db.commit()
                    chunks += 1
                    if chunks % 10 == 0:
                        print(f'CACHE chunks={chunks} episodes={count} rows={total}', flush=True)
                print(f'Cached {path.name}: {count} episodes, {total} rows', flush=True)
            if pending:
                raise ValueError('Training files end with an unfinished trajectory')
            if not count:
                raise ValueError('No complete multi-row trajectories found')
            db.execute('INSERT INTO metadata VALUES (?)', (pickle.dumps((mean, np.sqrt(m2/total)+1e-6)),))
            db.commit()
            db.close()
            temporary.replace(target)
        self.db = sqlite3.connect(str(target))
        self.state_mean, self.state_std = pickle.loads(self.db.execute('SELECT data FROM metadata').fetchone()[0])
        self.traj_lens = np.fromiter((r[0] for r in self.db.execute('SELECT length FROM episodes ORDER BY id')), dtype=np.int64)
        self.cumulative_lengths = np.cumsum(self.traj_lens)
        self.sorted_inds = range(len(self.traj_lens))
        self.trajectories = self

    def __len__(self):
        return len(self.traj_lens)

    def __getitem__(self, index):
        return pickle.loads(self.db.execute('SELECT data FROM episodes WHERE id=?', (int(index),)).fetchone()[0])

    def sample(self, size):
        from torch.utils.data._utils.collate import default_collate
        indices = np.searchsorted(self.cumulative_lengths, np.random.randint(self.cumulative_lengths[-1], size=size), side='right')
        return default_collate([EpisodeReplayBuffer.__getitem__(self, int(i)) for i in indices])
