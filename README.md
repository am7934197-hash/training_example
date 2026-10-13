# PI0.5 训练与逐 Action Dimension 误差分析

本目录用于训练 G1 place 任务的 PI0.5 相对动作模型，并分析模型在不同关节和不同预测时域上的 action chunk 误差。

除了原有的 flow-matching 训练 loss，还支持：

- 每个 action dimension 的 MAE 和 MSE；
- 形状为 `[action_dim, horizon]` 的 MAE/MSE 矩阵；
- action dimension × horizon step 的 MAE heatmap；
- 训练期间低频统计并上传 SwanLab；
- 使用 checkpoint 对数据集执行完整离线 replay。

普通在线推理没有 ground truth，不计算 action loss。逐维误差只在训练 batch 或离线 replay 数据上计算。

## 目录内容

| 文件 | 作用 |
| --- | --- |
| `pi05_train_relative.yaml` | 当前训练配置，包含数据、模型、优化器、SwanLab 和误差统计参数 |
| `pi05_train_relative.sh` | Accelerate 多卡启动脚本 |
| `recompute_relative_stats.py` | 创建数据集硬链接副本，并重新计算相对动作统计量 |
| `offline_replay_action_metrics.py` | 用 checkpoint 对离线数据计算逐维及逐 horizon 误差 |
| `logs/` | 训练日志和 PID 文件 |

误差累加器和训练循环接线位于当前 `lerobot_new` 环境实际加载的 LeRobot 源码：

```text
/home/zhangyuqi/zhangyuqi/from_jizheng/lerobotswanlab-swanlab/src/lerobot/
├── utils/action_error_metrics.py
├── scripts/lerobot_train.py
├── configs/train.py
└── common/wandb_utils.py
```

## 当前训练设置

`pi05_train_relative.yaml` 的主要参数如下：

| 参数 | 当前值 | 说明 |
| --- | ---: | --- |
| `policy.chunk_size` | 50 | 每次预测 50 个 horizon step |
| `policy.n_action_steps` | 50 | action chunk 长度 |
| `policy.max_action_dim` | 32 | 模型内部最大 action 维度；实际统计会裁到数据集 action 维度 |
| `policy.use_relative_actions` | `true` | 使用相对动作训练 |
| `policy.relative_exclude_joints` | `['gripper']` | 名称含 `gripper` 的维度保持绝对动作 |
| `batch_size` | 16 | 每张 GPU 的 batch size |
| `gradient_accumulation_steps` | 2 | 每 2 个 micro-batch 更新一次参数 |
| 有效 batch size | 64 | `16 × 2 GPUs × 2 accumulation` |
| `steps` | 40000 | 总训练 step |
| `save_freq` | 2000 | checkpoint 保存间隔 |
| `action_error_metrics.every_n_steps` | 500 | 完整 action chunk 误差统计间隔 |
| `action_error_metrics.heatmap_every_n_reports` | 4 | 每 4 次误差报告上传一次 heatmap，即每 2000 step |
| `wandb.backend` | `swanlab` | 使用 SwanLab 后端 |

## 训练前检查

进入本目录并确认 Python 加载的是已修改的 LeRobot 源码：

```bash
cd /home/zhangyuqi/zhangyuqi/G1-20260911/sorting_task/pi05_training_atciondim

conda run -n lerobot_new python -c \
  "import lerobot; print(lerobot.__file__)"
```

预期路径应位于：

```text
/home/zhangyuqi/zhangyuqi/from_jizheng/lerobotswanlab-swanlab/src/lerobot/
```

然后检查配置中的数据集和基础模型路径：

```bash
test -f /home/zhangyuqi/zhangyuqi/G1-20260911/G1_data/pick_place_9.11/only_pick/meta/info.json
test -d /home/zhangyuqi/zhangyuqi/models/models/pi05_base
```

如果第一条检查失败，需要先恢复数据集，或修改 `pi05_train_relative.yaml` 中的 `dataset.root`。

## 准备相对动作统计量

当前 `only_pick/meta/stats.json` 已包含 50-step 相对动作统计量：`action.count` 大于原始帧数，非夹爪 action 的均值接近 0，夹爪仍保持绝对值。因此当前训练不需要再执行本步骤。

只有换成尚未计算相对动作统计量的新数据集时，才需要执行：

```bash
conda run -n lerobot_new python recompute_relative_stats.py
```

调用脚本时应显式传入新数据集的源目录和输出目录，例如：

```bash
conda run -n lerobot_new python recompute_relative_stats.py \
  --source /path/to/absolute_action_dataset \
  --output /path/to/relative_stats_dataset
```

脚本会：

- 对 parquet 和视频优先使用硬链接，避免重复占用磁盘；
- 使用 `chunk_size=50`；
- 将普通关节转换为相对动作，gripper 维度保持绝对动作。

如果目标目录已经存在，脚本默认停止。只有确认需要替换时才使用 `--force`：

```bash
conda run -n lerobot_new python recompute_relative_stats.py --force
```

`--force` 会删除并重新创建目标数据集目录。

## 启动训练

### 当前推荐命令

当前目录中只有 `pi05_train_relative.yaml`。为避免启动脚本仍引用旧文件名的问题，建议直接执行：

```bash
cd /home/zhangyuqi/zhangyuqi/G1-20260911/sorting_task/pi05_training_atciondim

export SWANLAB_API_KEY="你的 SwanLab API Key"
export CUDA_VISIBLE_DEVICES=4,5
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

bash pi05_train_relative.sh
```

启动脚本会在真正启动前检查 GPU 进程数和有效 batch；当前应打印：

```text
Visible GPUs: 4,5; processes: 2
Effective batch size: 16 x 2 x 2 = 64
```

单卡示例：

```bash
export SWANLAB_API_KEY="你的 SwanLab API Key"
CUDA_VISIBLE_DEVICES=7 conda run -n lerobot_new accelerate launch \
  --num_processes=1 \
  --mixed_precision=bf16 \
  -m lerobot.scripts.lerobot_train \
  --config "$PWD/pi05_train_relative.yaml"
```

## 训练 loss 与 Action Replay 误差的区别

这两类指标含义不同，不能直接互相替代：

1. 原训练 loss 是 PI0.5 的 flow-matching loss，用于反向传播和更新模型。
2. Action replay 指标会额外执行一次无梯度的完整 action chunk 推理。
3. 模型输出经过反归一化和 relative-to-absolute 转换后，再与原始数据集 action 比较。
4. Replay 指标只用于诊断，不参与反向传播，也不会改变训练目标。

训练内的 replay 使用独立随机数生成器，因此不会消耗训练 loss 使用的随机序列。多卡训练会汇总所有进程上的预测和 target，只有主进程写文件和上传 SwanLab。

episode 结尾超出有效长度的 action padding 会被排除，避免后半段 horizon 被重复末帧污染。

## 训练阶段输出

结果保存到：

```text
<output_dir>/action_error_metrics/
├── train_replay_per_dim.csv
├── train_replay_mae_matrix.csv
├── train_replay_mse_matrix.csv
├── train_replay_mae_matrix.npy
├── train_replay_mse_matrix.npy
└── train_replay_mae_heatmap.png
```

各文件含义：

- `train_replay_per_dim.csv`：每个维度的 `action_dim`、`action_name`、MAE 和 MSE；
- `*_matrix.csv`：方便直接检查或导入表格软件；
- `*_matrix.npy`：保留纯数值矩阵，shape 为 `[action_dim, horizon]`；
- `*_mae_heatmap.png`：横轴为 horizon step `0~49`，纵轴为 action dimension/关节名称。

训练输出是当前进程生命周期内所有低频 replay batch 的累计平均。恢复训练时会开始一轮新的指标累计。

## SwanLab 指标

训练和离线 replay 使用相同的命名：

```text
replay/mae
replay/mse
replay/mae_dim_0
replay/mse_dim_0
replay/mae_dim_1
replay/mse_dim_1
...
replay/mae_heatmap
```

逐维标量每次误差报告都会记录。heatmap 不会每个 step 上传，默认每 4 次报告上传一次。

如果不希望训练期间执行额外 chunk 推理，可在 YAML 中关闭：

```yaml
action_error_metrics:
  enabled: false
```

如果只想降低开销，可增大 `every_n_steps`，或者减小 `num_inference_steps`。

## 离线 Replay 评估

离线 replay 必须提供 checkpoint 的 `pretrained_model` 目录，并显式传入当前存在的 YAML：

```bash
cd /home/zhangyuqi/zhangyuqi/G1-20260911/sorting_task/pi05_training_atciondim

conda run -n lerobot_new python offline_replay_action_metrics.py \
  --policy-path /home/zhangyuqi/zhangyuqi/models/g1/place/checkpoints/040000/pretrained_model \
  --config "$PWD/pi05_train_relative.yaml" \
  --output-dir /home/zhangyuqi/zhangyuqi/models/g1/place/offline_replay_metrics \
  --device cuda \
  --batch-size 8 \
  --log-every 20 \
  --heatmap-every 100 \
  --seed 42
```

常用参数：

| 参数 | 默认值 | 说明 |
| --- | ---: | --- |
| `--policy-path` | 必填 | checkpoint 的 `pretrained_model` 目录 |
| `--config` | 当前目录的 `pi05_train_relative.yaml` | 训练配置和数据预处理参数 |
| `--output-dir` | `offline_replay_metrics` | CSV、NPY 和 heatmap 输出目录 |
| `--device` | 自动选择 | 有 CUDA 时默认 `cuda`，否则为 `cpu` |
| `--batch-size` | 8 | replay batch size |
| `--num-workers` | 4 | DataLoader worker 数量 |
| `--max-batches` | 0 | 0 表示完整数据集，否则只评估指定 batch 数 |
| `--log-every` | 20 | 每多少个 replay batch 写文件并记录 SwanLab 标量 |
| `--heatmap-every` | 100 | 每多少个 replay batch 更新并上传 heatmap |
| `--num-inference-steps` | checkpoint 默认值 | PI0.5 去噪步数 |
| `--seed` | 42 | 固定 action sampling noise，便于重复比较 |
| `--no-swanlab` | 关闭 | 不初始化 SwanLab，只保存本地结果 |

快速检查前几个 batch：

```bash
conda run -n lerobot_new python offline_replay_action_metrics.py \
  --policy-path /home/zhangyuqi/zhangyuqi/models/g1/place/checkpoints/040000/pretrained_model \
  --config "$PWD/pi05_train_relative.yaml" \
  --output-dir /tmp/pi05_replay_smoke \
  --max-batches 5 \
  --batch-size 2 \
  --no-swanlab
```

离线输出文件名前缀为 `replay_`，其结构与训练输出一致；脚本结束时一定会保存最终 CSV、NPY 和 heatmap。

## 如何定位问题

建议按以下顺序查看：

1. 先比较 `replay/mae_dim_*`，定位整体误差最大的 action dimension。
2. 在 `replay_per_dim.csv` 中通过 `action_name` 确认对应关节。
3. 查看 heatmap，判断误差是从 horizon 0 就偏大，还是随预测时域逐渐增大。
4. 使用 `*_mae_matrix.npy` 做排序或自定义可视化。

加载矩阵示例：

```python
import numpy as np

mae = np.load("offline_replay_metrics/replay_mae_matrix.npy")
dim, horizon = np.unravel_index(np.argmax(mae), mae.shape)
print(f"最大 MAE: dim={dim}, horizon={horizon}, value={mae[dim, horizon]:.6f}")
```

一般来说：

- 某一维在所有 horizon 都高：优先检查该关节的数据范围、统计量、单位或 action 映射；
- 多数维度随 horizon 增长而变差：更可能是长时域预测误差累积；
- 只有末尾 horizon 异常：检查 episode 边界和 padding；本实现已自动忽略标准 `action_is_pad`；
- gripper 与其他维度量级差异明显：注意 gripper 保持绝对动作，其他普通关节使用相对动作。

## 常见问题

### 找不到 `lerobot.utils.action_error_metrics`

说明当前 Python 没有加载本项目修改过的 LeRobot。用以下命令确认：

```bash
conda run -n lerobot_new python -c "import lerobot; print(lerobot.__file__)"
```

### SwanLab 没有数据

检查：

```bash
echo "${SWANLAB_API_KEY:+SWANLAB_API_KEY is set}"
```

并确认 YAML 中：

```yaml
wandb:
  enable: true
  backend: swanlab
  mode: online
```

### GPU 显存不足

训练内 action replay 会额外执行完整的 10-step chunk 推理。可以依次尝试：

1. 增大 `action_error_metrics.every_n_steps`；
2. 减小 `action_error_metrics.num_inference_steps`；
3. 降低训练或离线 replay 的 batch size；
4. 暂时设置 `action_error_metrics.enabled: false`。
