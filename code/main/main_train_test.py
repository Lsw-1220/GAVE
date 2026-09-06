import os
import sys
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from run.run_decision_transformer import run_dt
import datetime
from pathlib import Path
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--step_num', type=int, default=400_000)
    parser.add_argument('--save_step', type=int, default=5_000)
    parser.add_argument('--train_csvs', '--dir', dest='train_csvs', nargs='+',
                        default=[f"D:/research/Experiment/dgab/data/MDP/trajectory/trajectory_data_{i}.csv" for i in (1, 2, 3)],
                        help='One or more training CSV paths; all rows are used for training.')
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--time_dim', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--expectile', type=float, default=0.99)
    parser.add_argument('--loss_report', type=int, default=100)
    parser.add_argument('--n_ctx', type=int, default=1024)
    parser.add_argument('--n_embd', type=int, default=512)
    parser.add_argument('--n_layer', type=int, default=8)
    parser.add_argument('--n_head', type=int, default=16)
    parser.add_argument('--n_inner', type=int, default=1024)
    parser.add_argument('--activation_function', type=str, default="relu")
    parser.add_argument('--n_position', type=int, default=1024)
    parser.add_argument('--resid_pdrop', type=float, default=0.1)
    parser.add_argument('--attn_pdrop', type=float, default=0.1)
    parser.add_argument('--budget_rate', type=float, default=1.0)
    parser.add_argument('--micro_batch_size', type=int, default=4)
    parser.add_argument('--train_cache_dir', default='./train_cache')
    parser.add_argument('--csv_chunksize', type=int, default=10000)
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Fixed checkpoint directory, useful for managed experiments.')
    args = parser.parse_args()

    model_param = {
        "micro_batch_size": args.micro_batch_size,
        "train_cache_dir": args.train_cache_dir,
        "csv_chunksize": args.csv_chunksize,
        "step_num": args.step_num,
        "save_step": args.save_step,
        "train_csvs": args.train_csvs,
        "hidden_size": args.hidden_size,
        "learning_rate": args.learning_rate,
        "time_dim": args.time_dim,
        "batch_size": args.batch_size,
        "device": args.device,
        "expectile": args.expectile,
        "loss_report": args.loss_report,
        "budget_rate": args.budget_rate,
        "block_config": {
            "n_ctx": args.n_ctx,
            "n_embd": args.n_embd,
            "n_layer": args.n_layer,
            "n_head": args.n_head,
            "n_inner": args.n_inner,
            "activation_function": args.activation_function,
            "n_position": args.n_position,
            "resid_pdrop": args.resid_pdrop,
            "attn_pdrop": args.attn_pdrop,
        }
    }

    print("Settings: \n", model_param)

    time_now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    log_dir = Path("./log")
    log_dir.mkdir(parents=True, exist_ok=True)
    filename = log_dir / f"gave_train_{time_now}"
    model_param['save_dir'] = args.save_dir or str(Path("./saved_model") / f"DTtest_{time_now}")
    with open(f"{filename}_param.txt", 'w', encoding='utf-8') as file:
        file.write(json.dumps(model_param, ensure_ascii=False, indent=2))

    print("##################Training##################")
    run_dt(device=model_param["device"], step_num=model_param["step_num"], dir=model_param["train_csvs"],
           save_step=model_param["save_step"], model_param=model_param, batch_size=model_param["batch_size"],
           save_dir=model_param['save_dir'], loss_report=model_param['loss_report'])

    print(f"Training complete. Checkpoints saved in: {model_param['save_dir']}")
