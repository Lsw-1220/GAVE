import os
import numpy as np
import pandas as pd
import pickle
import warnings

warnings.filterwarnings('ignore')


class TestDataLoader:
    def __init__(self, file_path="./data/log.csv", cache_dir=None):
        self.file_path = file_path
        cache_dir = cache_dir or os.path.dirname(file_path)
        os.makedirs(cache_dir, exist_ok=True)
        import hashlib
        import sqlite3
        stat = os.stat(file_path)
        signature = f'{os.path.abspath(file_path)}:{stat.st_size}:{stat.st_mtime_ns}:v1'
        key = hashlib.sha256(signature.encode()).hexdigest()[:20]
        target = os.path.join(cache_dir, key + '.sqlite')
        if not os.path.exists(target):
            temporary = target + '.building'
            if os.path.exists(temporary):
                os.remove(temporary)
            db = sqlite3.connect(temporary)
            columns = ['deliveryPeriodIndex', 'advertiserNumber', 'timeStepIndex',
                       'pValue', 'pValueSigma', 'leastWinningCost', 'budget',
                       'CPAConstraint', 'advertiserCategoryIndex']
            for chunk in pd.read_csv(file_path, usecols=columns, chunksize=10000):
                chunk.to_sql('traffic', db, if_exists='append', index=False)
            db.execute('CREATE INDEX episode ON traffic(deliveryPeriodIndex, advertiserNumber, timeStepIndex)')
            db.commit()
            db.close()
            os.replace(temporary, target)
        self.db = sqlite3.connect(target)
        self.keys = self.db.execute('SELECT DISTINCT deliveryPeriodIndex, advertiserNumber FROM traffic ORDER BY deliveryPeriodIndex, advertiserNumber').fetchall()
        if not self.keys:
            raise ValueError('Empty offline test dataset')
        self.test_dict = None

    def mock_data(self, key):
        data = pd.read_sql_query(
            'SELECT * FROM traffic WHERE deliveryPeriodIndex=? AND advertiserNumber=? ORDER BY timeStepIndex, rowid',
            self.db, params=key)
        pValues = data.groupby('timeStepIndex')['pValue'].apply(list).apply(np.array).tolist()
        pValueSigmas = data.groupby('timeStepIndex')['pValueSigma'].apply(list).apply(np.array).tolist()
        leastWinningCosts = data.groupby('timeStepIndex')['leastWinningCost'].apply(list).apply(np.array).tolist()
        num_timeStepIndex = len(pValues)
        budget = data['budget'].iloc[0]
        cpa = data['CPAConstraint'].iloc[0]
        category = data['advertiserCategoryIndex'].iloc[0]
        return num_timeStepIndex, pValues, pValueSigmas, leastWinningCosts, budget, cpa, category


if __name__ == '__main__':
    pass
