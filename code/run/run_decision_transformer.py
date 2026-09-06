import numpy as np
from bidding_train_env.common.utils import normalize_state, normalize_reward, save_normalize_dict
from bidding_train_env.baseline.dt.disk_buffer import DiskReplayBuffer
from bidding_train_env.baseline.dt.dt import GAVE
import torch
import logging
import pickle

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(name)s] [%(filename)s(%(lineno)d)] [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

def run_dt(device="cpu", step_num=10000, dir="./data/trajectory/trajectory_data.csv", save_step=5000, model_param=None,
           batch_size=32, save_dir="saved_model/DTtest", loss_report=2):
    train_model(device, step_num, dir=dir, save_step=save_step, model_param=model_param, batch_size=batch_size,
                save_dir=save_dir, loss_report=loss_report)


def train_model(device="cpu", step_num=10000, dir="./data/trajectory/trajectory_data.csv", save_step=5000, model_param=None,
                batch_size=32, save_dir="saved_model/DTtest", loss_report=2):
    if model_param is None:
        model_param = {}
    if step_num <= 0:
        raise ValueError("step_num must be greater than zero")
    if save_step <= 0:
        raise ValueError("save_step must be greater than zero")
    if str(device).startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable in this Python environment. Use a CUDA-enabled PyTorch environment or --device cpu.')
    if loss_report <= 0:
        raise ValueError('loss_report must be positive')
    state_dim=16
    replay_buffer = DiskReplayBuffer(16, 1, data_path=dir,
        cache_dir=model_param.get("train_cache_dir", "./train_cache"),
        chunksize=model_param.get("csv_chunksize", 10000))
    save_normalize_dict({"state_mean": replay_buffer.state_mean, "state_std": replay_buffer.state_std},
                        save_dir)
    logger.info(f"Replay buffer size: {len(replay_buffer.trajectories)}")

    model_param['state_mean'] = replay_buffer.state_mean
    model_param['state_std'] = replay_buffer.state_std
    model_param['device'] = device
    model = GAVE(state_dim=state_dim, act_dim=1,
                                hidden_size=model_param['hidden_size'], state_mean=model_param['state_mean'],
                                state_std=model_param['state_std'], device=model_param['device'],
                                learning_rate=model_param["learning_rate"], time_dim=model_param['time_dim'],
                                block_config=model_param['block_config'], expectile=model_param['expectile']
                                ).to(device)
    micro_batch_size = model_param.get('micro_batch_size', 4)
    if batch_size <= 0 or micro_batch_size <= 0:
        raise ValueError('Batch sizes must be positive')
    model.train()
    for step in range(step_num):
        # Collate only one effective batch on CPU; transfer one microbatch at a time.
        batch = replay_buffer.sample(batch_size)
        total_tokens = batch[7].sum().item()
        train_loss = np.zeros(9)
        for start in range(0, batch_size, micro_batch_size):
            end = min(start + micro_batch_size, batch_size)
            micro = tuple(t[start:end].to(device) for t in batch)
            weight = micro[7].sum().item() / total_tokens
            metrics = model.step(*micro, loss_scale=weight,
                                 zero_grad=(start == 0), update=(end == batch_size))
            train_loss += np.asarray(metrics) * weight
        i = step + 1
        if i%loss_report==0:
            logger.info("Step: {}, All loss: {}, loss1: {}, loss2: {}, loss3: {}, loss4: {}, w: {}, score_target: {}, score_preds: {}, score_preds1: {}"
                        .format(i, train_loss[0], train_loss[1], train_loss[2], train_loss[3], train_loss[4],
                                train_loss[5], train_loss[6], train_loss[7], train_loss[8]))
        model.scheduler.step()
        if i % save_step == 0:
            model.save_net(save_dir, "step_{}.pt".format(i))
    if i % save_step != 0:
        model.save_net(save_dir, "step_{}.pt".format(i))
    replay_buffer.db.close()
    test_state = np.ones(state_dim, dtype=np.float32)
    logger.info(f"Test action: {model.take_actions(test_state)}")


def load_model(device="cpu"):
    with open('./Model/DT/saved_model/normalize_dict.pkl', 'rb') as f:
        normalize_dict = pickle.load(f)
    model = GAVE(state_dim=16, act_dim=1, state_mean=normalize_dict["state_mean"],
                                state_std=normalize_dict["state_std"]).to(device)
    model.load_net("Model/DTtest/saved_model", device=device)
    test_state = np.ones(16, dtype=np.float32)
    logger.info(f"Test action: {model.take_actions(test_state)}")


if __name__ == "__main__":
    run_dt()
