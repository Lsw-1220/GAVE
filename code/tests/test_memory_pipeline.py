import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from bidding_train_env.baseline.dt.utils import EpisodeReplayBuffer
from bidding_train_env.baseline.dt.disk_buffer import DiskReplayBuffer
from bidding_train_env.baseline.dt.dt import GAVE
from run.checkpoint_selection import select_checkpoints


class PipelineTest(unittest.TestCase):
    def test_cache_and_accumulation(self):
        with tempfile.TemporaryDirectory() as folder:
            paths = []
            for file_index in range(3):
                rows = []
                for length in [3, 22]:
                    for t in range(length):
                        state = [float(t+file_index)/30]*16
                        rows.append(dict(state=str(state), next_state=str(state), action=0.2,
                                         reward=1., done=int(t==length-1), budget=100., CPAConstraint=2.))
                path = Path(folder)/f'{file_index}.csv'
                pd.DataFrame(rows).to_csv(path,index=False)
                paths.append(str(path))
            old = EpisodeReplayBuffer(16,1,paths)
            disk = DiskReplayBuffer(16,1,paths,folder,chunksize=2)
            np.testing.assert_allclose(old.state_mean,disk.state_mean)
            np.testing.assert_allclose(old.state_std,disk.state_std)
            for i in range(len(disk)):
                old_index = list(old.sorted_inds).index(i)
                random.seed(1); expected = old[old_index]
                random.seed(1); actual = EpisodeReplayBuffer.__getitem__(disk,i)
                for x,y in zip(expected,actual):
                    torch.testing.assert_close(x,y)
            batch = disk.sample(5)
            config = dict(n_ctx=64,n_embd=16,n_head=2,n_inner=32,n_layer=1,resid_pdrop=0.,attn_pdrop=0.)
            torch.manual_seed(1)
            a=GAVE(16,1,disk.state_mean,disk.state_std,hidden_size=16,block_config=config)
            torch.manual_seed(1)
            b=GAVE(16,1,disk.state_mean,disk.state_std,hidden_size=16,block_config=config)
            a.step(*batch)
            for start in range(0,5,2):
                micro=tuple(t[start:start+2] for t in batch)
                b.step(*micro,loss_scale=micro[7].sum().item()/batch[7].sum().item(),zero_grad=start==0,update=start==4)
            for x,y in zip(a.parameters(),b.parameters()):
                torch.testing.assert_close(x,y,atol=1e-7,rtol=1e-5)
            disk.db.close()
            reopened=DiskReplayBuffer(16,1,paths,folder)
            self.assertEqual(len(reopened),6)
            reopened.db.close()

    def test_offline_disk_loader(self):
        from bidding_train_env.dataloader.test_dataloader import TestDataLoader
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'traffic.csv'
            rows=[dict(deliveryPeriodIndex=7,advertiserNumber=a,timeStepIndex=t,
                       pValue=v,pValueSigma=.1,leastWinningCost=.2,budget=100,
                       CPAConstraint=2,advertiserCategoryIndex=1)
                  for a,t,v in [(2,1,.3),(1,0,.4),(2,0,.5),(1,1,.6)]]
            pd.DataFrame(rows).to_csv(path,index=False)
            loader=TestDataLoader(str(path),folder)
            self.assertEqual(loader.keys,[(7,1),(7,2)])
            n,pv,_,_,budget,cpa,_=loader.mock_data((7,2))
            self.assertEqual(n,2)
            np.testing.assert_allclose(pv,[[.5],[.3]])
            self.assertEqual((budget,cpa),(100,2))
            loader.db.close()

    def test_selection(self):
        with tempfile.TemporaryDirectory() as folder:
            for step in range(1000,41000,1000):
                (Path(folder)/f'step_{step}.pt').touch()
            calls=[]
            def evaluate(**kwargs):
                step=int(Path(kwargs['model_name']).stem[5:]);calls.append(step)
                return (-(step-23000)**2,0.,0.,0.)
            with patch('run.checkpoint_selection.run_test',side_effect=evaluate):
                result=select_checkpoints(dict(save_dir=folder,test_csv='unused.csv'),None,Path(folder)/'result.csv')
            self.assertEqual(result['best_coarse_step'],20000)
            self.assertEqual(result['best_step'],23000)
            self.assertEqual(calls[:4],[10000,20000,30000,40000])
            self.assertEqual(len(calls),len(set(calls)))
            self.assertEqual(set(calls),set(range(10000,31000,1000))|{40000})

if __name__=='__main__':
    unittest.main()
