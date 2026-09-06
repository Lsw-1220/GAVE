# 本地多文件训练与模型筛选

在 `nips-Gebidding-env` 中，从项目工作区运行：

```powershell
conda activate nips-Gebidding-env
python GAVE/code/main/main_train_test.py --micro_batch_size 4
```

默认读取 `D:/research/Experiment/dgab/data/MDP/trajectory/trajectory_data_1.csv`、`trajectory_data_2.csv`、`trajectory_data_3.csv`，训练 400000 次优化器更新，有效 batch size 128，每 5000 步保存 `step_5000.pt` 等模型，共 80 个 checkpoint。默认保持原网络结构和 FP32 精度；每个有效 batch 分成 32 个 microbatch，按有效 token 数加权累积梯度，每次有效 batch 只裁剪、更新参数和推进学习率一次。可调整 `--micro_batch_size`，不改变有效 batch size。

训练 CSV 每次读取 10000 行，轨迹保存至 SQLite 缓存，仅保留长度索引与统计量在内存。采样仍按轨迹长度加权、有放回抽样，窗口与归一化沿用原实现。缓存按照源文件路径、大小、修改时间识别并复用；可用 `--train_cache_dir` 指向空间充足的磁盘。单步轨迹沿用旧实现忽略；文件必须以完整轨迹结束，未结束的轨迹会报错，避免错误拼接不同文件。首次建立全量缓存需要扫描全部 CSV。

训练入口只训练并保存模型，不会在训练结束后启动离线评测。

1. 评估 step_10000、step_20000、…、step_400000，共 40 个模型。
2. 以平均 score 最大为准找到粗筛最佳步数 S，评估 S-10000、S-9000、…、S-1000 与 S+1000、…、S+10000；已测模型复用结果，训练边界外的模型跳过。
3. 输出全部已评估模型中的最佳模型。同分选择更早的步数。每个模型重置随机种子为 42。

结果逐模型写入 `code/log/*_result.csv`；相邻的 `*.best.json` 记录粗筛最佳步数、最终最佳步数、模型绝对路径及指标。检查点位于 `code/saved_model/DTtest_<时间戳>`。检查点是推理权重，不包含优化器状态，不支持精确断点续训。重新执行独立评估会重新评估选定模型。

已有训练模型可独立筛选：

```powershell
python GAVE/code/main/main_evaluate_checkpoints.py --model_dir <模型目录> --param_file <训练生成的_param.txt> --test_csv D:/research/Experiment/dgab/data/MDP/traffic/period-7.csv --result_file <结果.csv> --cache_dir <测试缓存目录>
```

资源验证：在 nips-Gebidding-env（PyTorch 2.8.0+cu126）和 RTX 4060 Laptop 8 GB 上，使用三个真实 CSV 中的 12 条轨迹、正式 512 维 8 层网络完成两次 batch 128 更新，microbatch 4，峰值 allocated 约 460 MiB、reserved 510 MiB。此为短程验证，不代表全量运行时间和系统内存峰值。单个未压缩权重文件约 106 MB，80 个约 8.5 GB，另需训练缓存空间。

验证命令：

```powershell
$env:PYTHONPATH='D:/research/Experiment/GAVE/GAVE/code'
python -m unittest discover -s GAVE/code/tests -v
```

覆盖分块/多文件数据与旧实现一致性、归一化一致性、关闭 dropout 时整批更新与不等长 microbatch 累积更新一致性、磁盘离线读取及两阶段筛选去重。
