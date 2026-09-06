"""Two-stage, reproducible offline checkpoint selection."""
import json
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from run.run_evaluate import run_test


def select_checkpoints(model_param, data_loader, result_file, coarse_interval=10000, fine_interval=1000, neighbors=10):
    checkpoints = {}
    for path in Path(model_param['save_dir']).glob('*.pt'):
        stem = path.stem.removeprefix('step_')
        if stem.isdigit():
            checkpoints[int(stem)] = path
    coarse = sorted(s for s in checkpoints if s > 0 and s % coarse_interval == 0)
    if not coarse:
        raise ValueError('No checkpoints at the coarse evaluation interval')
    result_file = Path(result_file)
    result_file.parent.mkdir(parents=True, exist_ok=True)
    results = {}
    if result_file.exists():
        for row in pd.read_csv(result_file).to_dict('records'):
            results[int(row['step'])] = row
    def evaluate(steps, stage):
        for step in steps:
            if step in results:
                continue
            random.seed(42)
            np.random.seed(42)
            torch.manual_seed(42)
            values = run_test(file_path=model_param['test_csv'], model_name=checkpoints[step].name,
                              model_param=model_param, data_loader=data_loader)
            if not np.all(np.isfinite(values)):
                raise ValueError(f'Nonfinite metrics at step {step}: {values}')
            results[step] = dict(step=step, stage=stage, **dict(zip(['score', 'score1', 'conversion', 'exceed'], values)))
            pd.DataFrame(results.values()).sort_values('step').to_csv(result_file, index=False)
    evaluate(coarse, 'coarse')
    best_coarse = max(coarse, key=lambda s: (results[s]['score'], -s))
    requested = [best_coarse + offset*fine_interval for offset in range(-neighbors, neighbors+1)]
    fine = sorted(s for s in requested if s in checkpoints and s > 0)
    missing = [s for s in requested if 0 < s <= max(checkpoints) and s not in checkpoints]
    if missing:
        raise ValueError(f'Missing fine checkpoints: {missing}')
    evaluate(fine, 'fine')
    best = max(results, key=lambda s: (results[s]['score'], -s))
    summary = dict(best_coarse_step=best_coarse, best_step=best,
                   best_checkpoint=str(checkpoints[best].resolve()), metric='score',
                   test_csv=str(Path(model_param['test_csv']).resolve()), seed=42, **{'metrics': results[best]})
    result_file.with_suffix('.best.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)
    return summary
