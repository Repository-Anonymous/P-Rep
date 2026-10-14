用户：

仔细阅读下le-wm 和代码 他vit 在训练他的数据之前有pretrain吗 是vjepa 还是其他

助手：

结论：**LeWM 的 ViT 在训练各任务数据之前没有做视觉预训练；既不是 V-JEPA，也不是 DINO、ImageNet 或其他预训练模型。它是随机初始化，然后与预测器一起在任务轨迹数据上端到端训练。**

代码证据：

- [config/train/model/lewm.yaml](/home/coder/le-wm/config/train/model/lewm.yaml:3) 明确配置：

  ```yaml
  encoder:
    _target_: stable_pretraining.backbone.utils.vit_hf
    size: tiny
    patch_size: 14
    image_size: ${img_size}
    pretrained: false
  ```

- [train.py](/home/coder/le-wm/train.py:85) 直接通过 Hydra 实例化整个模型，没有加载 V-JEPA 或其他 encoder 权重：

  ```python
  world_model = hydra.utils.instantiate(cfg.model)
  ```

- 优化器负责整个 `model`，包括 ViT encoder，而不是冻结 encoder：[train.py](/home/coder/le-wm/train.py:87)

  ```python
  "modules": "model"
  ```

- encoder 和 predictor 联合参与损失反向传播；target embedding 也没有 `detach` 或 EMA teacher：[train.py](/home/coder/le-wm/train.py:27)

  ```python
  emb = output["emb"]
  tgt_emb = emb[:, n_preds:]
  pred_emb = self.model.predict(ctx_emb, ctx_act)

  pred_loss = (pred_emb - tgt_emb).pow(2).mean()
  sigreg_loss = self.sigreg(emb.transpose(0, 1))
  ```

实际流程是：

```text
随机初始化 ViT-Tiny
        ↓
任务视频帧 → ViT → CLS token → projector → z_t
                                           ↓
动作 → action encoder → predictor → 预测 z_(t+1)
                                           ↓
                prediction loss + SIGReg
                                           ↓
        ViT、projector、predictor 一起更新
```

这里的 ViT 是普通的 **ViT-Tiny/14**：

- 输入：224×224 图像
- patch size：14
- embedding dimension：192
- 使用 `[CLS]` token 作为图像表示，见 [jepa.py](/home/coder/le-wm/jepa.py:34)
- 不是 V-JEPA encoder，也没有采用 V-JEPA checkpoint

需要特别区分三个容易混淆的地方：

1. `stable_pretraining` 只是训练框架/代码库的名字  
   它并不意味着模型用了 pretrained weights。真正决定权重初始化的是 `pretrained: false`。

2. README 中的 “Pretrained LeWM checkpoints”  
   指的是作者已经在 PushT、Cube、TwoRooms、Reacher 等任务数据上训练完成的 **LeWM 成品 checkpoint**，不是 LeWM 启动训练时使用的视觉预训练权重。

3. `ckpt_path` 的加载  
   [train.py](/home/coder/le-wm/train.py:132) 会在同一个输出目录已经存在 checkpoint 时续训。这只是恢复之前的 LeWM 训练，不是 V-JEPA 预训练。

论文也把这一点作为核心贡献：LeWM 从原始像素端到端训练，不依赖 frozen/pretrained encoder、EMA teacher 或 stop-gradient；其防坍塌机制是 **SIGReg 高斯正则化**。[论文项目页](https://le-wm.github.io/)

所以一句话概括：

> **LeWM 借用了 JEPA 的“在表征空间预测未来”思想，但它的 ViT-Tiny 是从零开始训练的，并没有先用 V-JEPA 预训练。**

用户：

1. 他是 image vit 还是 video 的 vit 2. 他pretrain是拿image还是 video去做pretrain 3. 他pretrain的时候有像v-jepa 2 一样 mask住video吗

助手：

先纠正一个关键前提：**LeWM 没有单独的 pretrain 阶段**。作者直接在每个控制任务的轨迹数据上，从随机初始化开始端到端训练。

### 1. 它是 Image ViT 还是 Video ViT？

是 **Image ViT（ViT-Tiny/14）**，不是 V-JEPA 那种时空 Video ViT。

代码把视频序列的时间维展平到 batch：

```python
# pixels: (B, T, C, H, W)
pixels = rearrange(pixels, "b t ... -> (b t) ...")
output = self.encoder(pixels)
pixels_emb = output.last_hidden_state[:, 0]
```

见 [jepa.py](/home/coder/le-wm/jepa.py:34)。

因此每一帧都被 ViT **独立编码**：

```text
frame_t     → Image ViT → z_t
frame_t+1   → Image ViT → z_t+1
frame_t+2   → Image ViT → z_t+2
```

Image ViT 内部没有跨帧时空注意力。时间动态由后面的 action-conditioned autoregressive predictor 学习：

```text
历史 latent + 动作 → temporal predictor → 下一帧 latent
```

所以整体模型处理视频轨迹，但其中的视觉 encoder 本身仍是逐帧 Image ViT。

---

### 2. 它 pretrain 时使用 image 还是 video？

严格来说，**它没有预训练**：

```yaml
pretrained: false
```

见 [config/train/model/lewm.yaml](/home/coder/le-wm/config/train/model/lewm.yaml:8)。

它直接使用任务采集的连续轨迹训练，例如：

- PushT
- OGBench Cube
- TwoRoom
- Reacher

训练样本是连续帧和对应动作：

```text
(o_t, a_t), (o_t+1, a_t+1), ...
```

所以：

- 对 **encoder** 来说：输入是逐帧 image。
- 对 **整个 LeWM** 来说：训练数据是 video/trajectory sequence。
- 但这不是“先用视频预训练，再在任务数据微调”，而是直接在任务轨迹上训练 world model。

论文默认每个样本取 4 帧，并使用 frame-skip 5。前三个 latent 作为历史，根据相应动作预测后续 latent。

---

### 3. 训练时会像 V-JEPA 2 一样 mask video 吗？

**不会。LeWM 完全没有 video masking。**

配置明确关闭 mask token：

```yaml
use_mask_token: false
```

见 [config/train/model/lewm.yaml](/home/coder/le-wm/config/train/model/lewm.yaml:9)。

代码中也没有：

- spatial block masking
- temporal masking
- tube masking
- context/target mask selection
- masked-token prediction

所有输入帧都是完整图像。其目标是根据历史 latent 和动作预测下一时刻 latent：

```python
ctx_emb = emb[:, :ctx_len]
ctx_act = act_emb[:, :ctx_len]

tgt_emb = emb[:, n_preds:]
pred_emb = self.model.predict(ctx_emb, ctx_act)

pred_loss = (pred_emb - tgt_emb).pow(2).mean()
```

见 [train.py](/home/coder/le-wm/train.py:32)。

两者区别可以概括为：

| | LeWM | V-JEPA / V-JEPA 2 |
|---|---|---|
| 视觉 encoder | 逐帧 Image ViT | 时空 Video ViT |
| 初始化 | 随机初始化 | 大规模视频预训练 |
| 训练数据 | 特定控制任务轨迹 | 大规模视频数据 |
| masking | 没有 | mask 时空区域/tubes |
| 预测目标 | 下一时刻完整帧的 latent | 被 mask 视频区域的 latent |
| 动作条件 | 有 | 原始 V-JEPA 没有；action-conditioned 版本有 |
| 防坍塌 | SIGReg | EMA target encoder 等机制 |

因此最准确的描述是：

> **LeWM 不是先预训练一个 V-JEPA 视频编码器。它把随机初始化的 Image ViT 逐帧应用到控制轨迹上，再利用动作条件时序预测器预测下一帧表示；整个训练过程不做视频 masking。**

用户：

把我们俩的对话原封不动不动的同步到 /home/coder/P-Rep/le-wm/zeyu-understand.md

### V-JEPA 2.1 ViT-B 能接收多少帧？

助手：

1. V-JEPA 2.1 ViT-B 能接收多少帧？

官方标准配置一次接收 **64 帧视频**：

- `num_frames: 64`
- `tubelet_size: 2`
- 64 帧会形成 32 个时间 tubelets

见 [backbones.py](/home/coder/P-Rep/vjepa2/src/hub/backbones.py:210)。

训练过程则是：

- 主预训练使用 16 帧视频：[pretrain-256px-16f.yaml](/home/coder/P-Rep/vjepa2/configs/train_2_1/vitb16/pretrain-256px-16f.yaml:20)
- cooldown 阶段使用 64 帧视频：[cooldown-256px-64f.yaml](/home/coder/P-Rep/vjepa2/configs/train_2_1/vitb16/cooldown-256px-64f.yaml:19)
- 同时还支持 `T=1` 的单张图片输入

因为使用 RoPE 插值，它不完全锁死为 64 帧，通常也可以输入 16、32 等偶数帧；但官方这份模型最标准、经过训练验证的最大视频窗口是 64 帧。

不过需要特别注意：我们刚才替换进 LeWM 的实现，实际每次只给 V-JEPA **1 帧**，并没有给它一段视频。

原因是 LeWM 会先把时间维展平：

```python
pixels = rearrange(pixels, "b t ... -> (b t) ...")
```

见 [jepa.py](/home/coder/P-Rep/le-wm/jepa.py:34)。随后适配器把每张图片变成 `T=1` 输入：

```python
patch_tokens = self.backbone(pixels.unsqueeze(2))
```

见 [vjepa2_encoder.py](/home/coder/P-Rep/le-wm/vjepa2_encoder.py:62)。

所以目前实验是：

> V-JEPA 2.1 的 image pathway，逐帧编码；不是 16/32/64 帧联合编码的视频 pathway。

2. 43、44、45 有没有对应的 observation 视频？

**有对应 observation 数据。** 我直接检查了 HDF5，三个 example 都有连续 26 帧：

| Seed | Episode | observation 行 | step 范围 | 数据形状 |
|---|---:|---:|---:|---|
| 43 | 5053 | 1015694 开始 | 41–66 | `(26,224,224,3)` |
| 44 | 6668 | 1340356 开始 | 88–113 | `(26,224,224,3)` |
| 45 | 9245 | 1858300 开始 | 55–80 | `(26,224,224,3)` |

现有推理 MP4 是三栏组合视频：

```text
agent | dataset | goal
```

其中：

- 左边 `agent`：MuJoCo 实际推理轨迹
- 中间 `dataset`：对应的 26 帧 observation/expert trajectory
- 右边 `goal`：目标帧

三个视频分别是：

- [Seed 43 视频](/home/coder/P-Rep/le-wm/outputs/cube_vjepa2_3examples_20260812_174359/example_1_gpu0_seed43/cube_vjepa2_inference.mp4)
- [Seed 44 视频](/home/coder/P-Rep/le-wm/outputs/cube_vjepa2_3examples_20260812_174359/example_2_gpu1_seed44/cube_vjepa2_inference.mp4)
- [Seed 45 视频](/home/coder/P-Rep/le-wm/outputs/cube_vjepa2_3examples_20260812_174359/example_3_gpu2_seed45/cube_vjepa2_inference.mp4)

因此准确地说：**observation 视频有，但目前嵌在组合 MP4 的中间栏，没有另外输出独立的 observation-only MP4。**

用户：

把最近这两个对话 从 1. V-JEPA 2.1 ViT-B 能接收多少帧？  开始 ，记录在 /home/coder/P-Rep/le-wm/zeyu-understand.md

不是“data 43 只有 26 帧”。准确情况是：

```text
seed=43（随机种子）
    ↓ 随机抽到
episode=5053（完整 201 帧）
    ↓ 从 step 41 开始，goal_offset=25
本次 observation/reference 窗口 = step 41～66，共 26 帧
```

具体数据：

- 整个 Cube 数据集：`2,010,000` 帧、`10,000` 个 episode
- seed 43 不是数据编号，而是评估采样的随机种子
- 它抽到的是 `episode 5053`
- episode 5053 完整范围：
  - HDF5 行：`1015653～1015853`
  - step：`0～200`
  - 总计：**201 帧**
- 本次评估从：
  - start observation：step 41
  - goal observation：step 66
  - 中间连续窗口：`41～66`
  - 包含首尾，所以是 `66 - 41 + 1 = 26` 帧

为什么是 26 帧？

配置里是：

```yaml
goal_offset_steps: 25
```

代码读取：

```python
start_step : start_step + goal_offset + 1
```

见 [eval.py](/home/coder/P-Rep/le-wm/eval.py:118) 和 [world.py](/home/coder/P-Rep/le-wm/.venv/lib/python3.10/site-packages/stable_worldmodel/world/world.py:567)。

所以中间的 `dataset` observation 视频只是从完整 episode 中截取的评估片段：

```text
完整 episode 5053：
step 0 ───────── step 41 ═════════ step 66 ───────── step 200
                         本次使用的 26 帧
                         ↑初始帧          ↑目标帧
```

因此，截图里的表格更准确的标题应该是：

> seed 43/44/45 本次评估截取的 observation 窗口均为 26 帧。

而不是说整个 episode 或整个 data 只有 26 帧。

用户：

V-JEPA 的 64 帧会产生 32 个 temporal tubelet 表示，而 LeWM 的 predictor 只接受“当前状态”的单个 192 维 latent。为保持现有 predictor 可运行，我会对最后一个 temporal tubelet 做空间池化，再用当前固定 768→192 投影作为当前状态； 我做了 2个pooling 一个是 temporal tubelet 一个是 768 到 192 的 mean pool 这个显然是临时之举，根据这个两个问题 1 v-jepa 2的 predictor结构是怎么适应这个维度的 /home/coder/P-Rep/vjepa2？ 2. 他有没有release predictor的pretrain weight？

助手：

先纠正一下我们当前实现：并不是做了两次 mean pooling。

当前临时适配实际是：

```text
V-JEPA 输出：
[B, 18432, 768]
      ↓ reshape
[B, 32 temporal tubelets, 576 spatial patches, 768]
      ↓ 选择最后一个 temporal tubelet
[B, 576, 768]
      ↓ 对 576 个 spatial patches 做 mean pooling
[B, 768]
      ↓ 固定随机正交线性投影，不是 mean pooling
[B, 192]
```

对应代码在 [vjepa2_encoder.py](/home/coder/P-Rep/le-wm/vjepa2_encoder.py:59)。

所以临时之处主要有两个：

- 丢弃前 31 个 temporal tubelet，只保留最后一个
- 用未经训练的固定 `768→192` 正交投影适配 LeWM

## 1. V-JEPA 2.1 predictor 怎样处理这些维度？

V-JEPA 2.1 自己完全不需要把视频压缩成单个 latent。它一直保留时空 token。

对于 `64×384×384` 输入：

```text
64 frames / tubelet_size 2 = 32 temporal positions
384 / patch_size 16 = 24 spatial positions
32 × 24 × 24 = 18,432 tokens
```

因此 encoder 输出：

```text
[B, 18432, 768]
```

V-JEPA predictor 的处理流程是：

```text
context tokens: [B, N_context, 768]
        ↓ 可训练 predictor_embed
[B, N_context, 384]
        ↓
加入 target 位置的 mask tokens: [B, N_target, 384]
        ↓
根据 masks_x / masks_y 恢复时空顺序
        ↓
12 层 Predictor Transformer + 3D RoPE
        ↓ 可训练 predictor_proj
[B, N_target, 1664]
```

关键点如下。

### 它不 pool temporal tubelets

32 个 temporal tubelets 和每个 tubelet 的 576 个空间 patches 都保留为独立 token。

Predictor 使用 `masks_x` 指明可见 context tokens，使用 `masks_y` 指明需要预测的 target tokens：

```python
x = self.predictor_embed(x)
pred_tokens = apply_masks(mask_tokens, masks_y)
x = torch.cat([x, pred_tokens], dim=1)
```

见 [predictor.py](/home/coder/P-Rep/vjepa2/app/vjepa_2_1/models/predictor.py:215)。

然后 predictor 根据 token 的时空位置排列它们，并通过 Transformer attention 和 3D RoPE 建模跨时间、跨空间关系。

### 768→384 是训练出来的线性映射

ViT-B encoder token 是 768 维，predictor 内部宽度是 384：

```python
self.predictor_embed = nn.Linear(768, 384)
```

见 [predictor.py](/home/coder/P-Rep/vjepa2/app/vjepa_2_1/models/predictor.py:82)。

实际 checkpoint 参数形状：

```text
predictor_embed.weight: (384, 768)
predictor_embed.bias:   (384,)
```

这不是 pooling，而是预训练期间一起学习的投影。

### 输出为什么是 1664，不是 768？

这份模型叫：

```text
vjepa2_1_vitb_dist_vitG_384
```

它是从 ViT-G teacher 蒸馏的 ViT-B。Teacher 特征维度是 1664，因此输出投影为：

```text
predictor_proj: 384 → 1664
```

实际 checkpoint 形状：

```text
predictor_proj.weight: (1664, 384)
predictor_proj.bias:   (1664,)
```

配置位置见 [backbones.py](/home/coder/P-Rep/vjepa2/src/hub/backbones.py:288)。

因此，V-JEPA predictor 解决的是：

> 如何将大量 768 维时空 context tokens 转成预测位置上的 1664 维 teacher target tokens。

它并没有解决：

> 如何将整个视频变成 LeWM 所需的单个 192 维当前状态。

这两个 predictor 的接口本质不同：

| | V-JEPA 2.1 predictor | LeWM predictor |
|---|---|---|
| 输入 | masked 时空 patch tokens | 状态 latent + action |
| token 数 | 保留大量时空 tokens | 每个时刻一个 latent |
| 内部维度 | 384 | 192 |
| 输出 | 每个 target patch 的 1664D 特征 | 下一状态的 192D latent |
| action-conditioned | 否 | 是 |
| 目的 | masked latent prediction | 控制和 MPC rollout |

## 2. 是否发布了 predictor 的预训练权重？

**发布了，而且就在我们已经下载的同一个 checkpoint 里。**

文件：

[官方 V-JEPA 2.1 ViT-B checkpoint](/home/coder/P-Rep/le-wm/.stable-wm/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt)

其顶层包含：

```text
encoder
ema_encoder
predictor
opt
scaler
epoch
loss
...
```

其中 `predictor` 包含 162 个参数项，包括：

```text
predictor_embed
8 个 mask_tokens
12 层 predictor_blocks
predictor_norm
predictor_proj
predictor_proj_context
video_mod_embed
img_mod_embed
```

我已经使用仓库中的官方构造函数做了严格加载验证：

```text
All keys matched successfully
Predictor parameters: 22,874,752
Predictor depth: 12
Input projection:  768 → 384
Output projection: 384 → 1664
Mask tokens: 8
Maximum token grid: 18,432
```

官方加载代码也明确读取：

```python
predictor_state_dict = state_dict["predictor"]
predictor.load_state_dict(predictor_state_dict, strict=True)
```

见 [backbones.py](/home/coder/P-Rep/vjepa2/src/hub/backbones.py:272)。

不过，我们当前的 [vjepa2_encoder.py](/home/coder/P-Rep/le-wm/vjepa2_encoder.py:44) 只加载了：

```python
checkpoint["ema_encoder"]
```

还没有使用 checkpoint 里的 `predictor`。

## 能不能直接把官方 predictor 接到 LeWM？

不能直接接，主要不是维度问题，而是任务接口不同：

```text
V-JEPA predictor:
context tokens + mask locations
    → masked target patch features

LeWM predictor:
current latent + robot actions
    → future state latent
```

标准 V-JEPA 2.1 predictor 完全不接 action。即使加载其预训练权重，也不能让 CEM 比较不同候选动作。

仓库另外发布了 V-JEPA 2-AC 的 action-conditioned predictor：

- checkpoint：`vjepa2-ac-vitg.pt`
- 代码：[ac_predictor.py](/home/coder/P-Rep/vjepa2/src/models/ac_predictor.py)
- 加载入口：`vjepa2_ac_vit_giant`

但它基于 ViT-g，而不是现在的 V-JEPA 2.1 ViT-B；其机器人状态和 action 接口也不一定与 Cube 的 5 维 action 对齐，因此同样不能无训练地直接替换。

最合理的正式方案是：

```text
V-JEPA 2.1 ViT-B 的完整时空 tokens
        ↓
可训练的 attentive pooling / learned query
        ↓
192D state latent
        ↓
使用 Cube observation + action 重新训练 LeWM predictor
```

这样可以保留 V-JEPA 的 32 个 temporal tubelet 信息，同时消除目前“只选最后一个 tubelet”和“随机 768→192 投影”这两个临时适配。

用户：

就按照这个方案 1 重构一个predictor 不破坏lewm原来的pred 2. 看一下 lewm怎么训练的cube 这 10k 个episode怎么训练用的？

助手：

已经按方案完成独立重构，并没有修改 LeWM 原 predictor 的代码、配置或 checkpoint。

## 1. 新 V-JEPA predictor 路径

新结构为：

```text
每个控制状态之前的 64 帧
    ↓ V-JEPA 2.1 ViT-B（冻结）
18,432 × 768 时空 tokens
    ↓ learned-query attention pooling
单个 192D state latent
    ↓ 新的 VJEPAActionPredictor
历史 3 个 state latent + 3 个 action block
    ↓
预测后续 192D state latent
```

与之前的临时实现相比：

- 不再只取最后一个 temporal tubelet
- 不再进行 spatial mean pooling
- 不再使用随机固定 `768→192` 投影
- learned query 对全部 `32×24×24=18,432` 个时空 token 做 attention
- `768→192` 的 key/value 映射和 query 都参与 Cube 训练
- predictor 是全新初始化、单独保存的参数
- V-JEPA backbone 默认冻结

代码：

- [新 V-JEPA LeWM 模型](/home/coder/P-Rep/le-wm/vjepa2_lewm.py:24)
- [独立 action predictor](/home/coder/P-Rep/le-wm/vjepa2_lewm.py:53)
- [64 帧 Cube 数据集](/home/coder/P-Rep/le-wm/cube_video_context_dataset.py:13)
- [训练入口](/home/coder/P-Rep/le-wm/train_vjepa2_cube.py:20)
- [训练配置](/home/coder/P-Rep/le-wm/config/train/vjepa2_cube.yaml)
- [启动脚本](/home/coder/P-Rep/le-wm/run_train_vjepa2_cube.sh)

原文件 [module.py](/home/coder/P-Rep/le-wm/module.py:244) 和 [原 LeWM 模型配置](/home/coder/P-Rep/le-wm/config/train/model/lewm.yaml) 均无改动。

## 2. 原 LeWM 怎么使用 10k Cube episodes？

Cube 数据集包含：

```text
10,000 episodes
每个 episode 201 帧
合计 2,010,000 帧
```

原配置是：

```yaml
history_size: 3
num_preds: 1
num_steps: 4
frameskip: 5
```

见 [ogb.yaml](/home/coder/P-Rep/le-wm/config/train/data/ogb.yaml) 和 [lewm.yaml](/home/coder/P-Rep/le-wm/config/train/lewm.yaml)。

每个训练样本实际读取连续 20 个原始 step：

```text
raw rows:
start ... start+19

图像 observation:
start, start+5, start+10, start+15
共 4 帧

action:
[start:start+5]   → 展平为 25D
[start+5:start+10] → 展平为 25D
[start+10:start+15] → 展平为 25D
[start+15:start+20] → 展平为 25D
```

Cube 单步 action 是 5D，因此连续 5 个 action 展平后是：

```text
5 steps × 5D = 25D action block
```

每个 201 帧 episode 可以产生：

```text
201 - 20 + 1 = 182 个滑动窗口
```

全部数据产生：

```text
10,000 × 182 = 1,820,000 个训练样本
```

这些窗口高度重叠：

```text
window 0: raw step 0–19
window 1: raw step 1–20
window 2: raw step 2–21
...
```

### 原 train/val 划分

原代码先创建全部 1,820,000 个窗口，再随机做 90/10 划分：

```text
train: 1,638,000 windows
val:     182,000 windows
```

这意味着它是窗口级划分，不是 episode 级划分。同一个 episode 的相邻窗口可能分别进入 train 和 validation。

原训练设置：

- batch size：128
- 每 epoch 约 12,796 个 train batch
- epochs：100
- AdamW
- learning rate：`5e-5`
- weight decay：`1e-3`
- BF16
- ViT-Tiny 从随机初始化开始，与 predictor 端到端共同训练

### 原 loss

4 个状态 embedding：

```text
z0, z1, z2, z3
```

使用前三个状态和 action：

```text
(z0,a0), (z1,a1), (z2,a2)
```

预测：

```text
z1, z2, z3
```

损失是：

```text
prediction MSE + 0.09 × SIGReg
```

对应 [train.py](/home/coder/P-Rep/le-wm/train.py:14)。

## 新数据管线如何适配 64 帧？

为了避免把未来帧泄漏给 predictor，新样本是：

```text
state 0: 视频 [t-63 : t]
state 1: 视频 [t-58 : t+5]
state 2: 视频 [t-53 : t+10]
state 3: 视频 [t-48 : t+15]
```

每个 state clip 都只有截至该状态的历史，不包含它后面的预测目标。

由于 episode 开头不足 64 帧，每集可用起点从 step 63 开始。还要为四个 action block 留出 20 步，因此每集产生：

```text
state_start = 63 ... 181
共 119 个样本
```

新数据量：

```text
10,000 × 119 = 1,190,000 samples
```

新实现改成 episode 级划分，避免相邻窗口泄漏：

```text
train: 9,000 episodes，1,071,000 samples
val:   1,000 episodes，  119,000 samples
```

## 验证情况

完成了真实 `64帧 × 4状态` 前向和反向：

```text
state embedding: (1,4,192)
action embedding: (1,4,192)
prediction: (1,3,192)
```

验证结果：

- loss 有限
- V-JEPA backbone：无梯度，保持冻结
- learned pooler：全部参数有梯度
- 新 predictor：全部参数有梯度
- 可训练参数：约 13.1M
- smoke test 峰值显存：约 1.8 GiB

Lightning 完整训练链路也已完成 2 个 train step 和 validation：

```text
train loss: 0.2059 → 0.2083
val loss:   0.0823 → 0.0742
```

独立 smoke checkpoint：

- [epoch 1](/home/coder/P-Rep/le-wm/.stable-wm/checkpoints/cube-vjepa2-lewm-smoke/weights_epoch_1.pt)
- [epoch 2](/home/coder/P-Rep/le-wm/.stable-wm/checkpoints/cube-vjepa2-lewm-smoke/weights_epoch_2.pt)

尚未启动完整 100-epoch 训练；当前只完成了结构、数据和优化链路验证。正式启动命令：

```bash
cd /home/coder/P-Rep/le-wm
CUDA_VISIBLE_DEVICES=0 ./run_train_vjepa2_cube.sh
```

---

## 原版 LeWM Cube 与我们的推荐训练超参数

### 用户

按照上面的表格 在右边在加一列 写我们的推荐超参

### 助手

按“复现原版 LeWM Cube 训练”的目标，我们推荐在当前服务器上使用 4 张 RTX PRO 6000，并保持 global batch size=128：

| 项目 | 原版 Cube 设置 | 我们的推荐超参 |
|---|---:|---:|
| GPU 数量 | 1 张 | **4 张** |
| GPU 型号 | 未披露 | **RTX PRO 6000 Blackwell 96GB** |
| 每卡 Batch size | 128 | **32** |
| Global batch size | 128 | **128** |
| 优化器 | AdamW | **AdamW** |
| Learning rate | 5 × 10⁻⁵ | **5 × 10⁻⁵** |
| Weight decay | 1 × 10⁻³ | **1 × 10⁻³** |
| 训练轮数 | 10 epochs | **先跑 10 epochs** |
| Optimizer iterations | 约 127,960 | **约 127,960** |
| 精度 | bf16 | **bf16** |
| Gradient clipping | 1.0 | **1.0** |
| 输入分辨率 | 224 × 224 | **224 × 224** |
| Observation history | 3 | **3** |
| Prediction steps | 1 | **1** |
| Frameskip | 5 | **5** |
| 单样本 observation 数 | 4 | **4** |
| Train/val split | 90% / 10% | **90% / 10%** |
| DataLoader workers | 6 | **每卡 6，共 24** |
| Scheduler | Linear warmup + cosine | **保持一致** |
| SIGReg weight | 0.09 | **0.09** |
| 随机种子 | 3072 | **3072，并额外跑 2 个种子验证** |
| Checkpoint | 未明确 | **每 epoch 保存，按 val loss 取 best** |

最关键的是：Lightning DDP 中配置里的 `batch_size` 是每张 GPU 的 batch size。因此使用 4 卡时必须从 `128` 改成 `32`：

```yaml
trainer:
  accelerator: gpu
  devices: 4
  precision: bf16-mixed
  max_epochs: 10
  gradient_clip_val: 1.0

loader:
  batch_size: 32
  num_workers: 6
```

这样：

```text
4 GPUs × 32 samples/GPU = global batch size 128
```

global batch 和原版保持不变，所以学习率仍用 `5e-5`，不需要做线性放大。每个 epoch 的 optimizer step 数也仍约为 `12,796`，10 epochs 约为 `127,960`。

这套推荐针对的是原版 LeWM Cube。后续若训练“64 帧 V-JEPA 2.1 ViT-B + 新 predictor”，显存和计算量会大很多，不能直接沿用每卡 batch 32，需要单独做 batch-size profiling。

---

## ogbench

### 回答一

看了 [PDF 第 21 页的 Table 5](/home/coder/P-Rep/ogbench/ogbench.pdf) 和本地实现。**LeWM 可以扩展到这些机械臂任务，但现有模型并没有全部支持；原版 LeWM 的输入是图像，不能直接吃 state 向量。**

Table 5 全部都是机械臂任务：

| 类别 | 环境变体 | 主要能力 | 当前 LeWM 的情况 |
|---|---|---|---|
| Cube | single、double、triple、quadruple | 抓取、搬运、堆叠、交换方块 | single 已跑通；多方块需要适配目标设置并训练 |
| Scene | scene | 按按钮解锁、操作抽屉/窗户、移动方块 | 已安装的环境库有 Scene，但仓库没有对应 LeWM 配置和权重 |
| Puzzle | 3×3、4×4、4×5、4×6 | 按按钮翻转相邻颜色，完成组合目标 | 需要接入环境、训练模型，并验证长序列规划 |

共 **9 种环境 × state/visual × play/noisy = 36 组数据集**。

**带不带 `visual-`，区别在模型能看到什么：**

| 项目 | 不带 visual | 带 visual |
|---|---|---|
| 观测与目标 | 数值状态：关节、夹爪、物体位置、按钮状态等 | **64×64 RGB 图像**，目标也是图像 |
| 渲染 | 普通外观 | 调整颜色、使用视觉相机、机械臂半透明，减少遮挡 |
| 控制 | 两者都是相同的 5 维连续动作：末端 xyz、yaw、夹爪开合 | 相同 |

所以 visual 主要改变观测方式和可见性，任务规则、控制接口基本相同。[官方环境定义](https://github.com/seohongpark/ogbench/blob/master/ogbench/manipspace/envs/manipspace_env.py)

`play/noisy` 是另一个独立维度：play 的轨迹更自然、噪声有时间相关性；noisy 使用更大的独立噪声，状态覆盖更广。它不是图像加噪声。

**对 LeWM，要区分“能适配”和“现成能做”：**

- **visual：符合原版 LeWM 的输入设计**，但各环境仍需相应数据训练，现有 `cube-lewm` 权重不能视为通用机械臂模型。
- **state：需要换成状态编码器并重新训练。**也可以把模拟器状态渲染成图片使用 LeWM，但这样评估的是视觉策略。
- LeWM 论文在 OGBench 上**只实验了单方块 Cube**，没有验证多方块、Scene、Puzzle 的全覆盖能力。[LeWM 论文附录 E](https://arxiv.org/html/2603.19312v1)

还有一个容易混淆的地方：当前 [cube.yaml](/home/coder/P-Rep/le-wm/config/eval/cube.yaml:6) 虽然写着 `ob_type: states`，但外层另外生成 **224×224 pixels**，[LeWM 编码器](/home/coder/P-Rep/le-wm/.venv/lib/python3.10/site-packages/stable_worldmodel/wm/lewm/lewm.py:25) 实际读取的是图片。因此之前跑通的是图像推理流程。

如果我们主要比较 **LeWM 和 V-JEPA 的视觉世界模型**，建议先做 visual 的 9 组 play，再补 noisy。要和 Table 5 对比，还需调整评估：**Table 5 使用每个环境的 5 个官方目标；当前脚本使用轨迹未来帧作为目标，两者成功率不能直接对比。**

### 回答二

**当前这套流程直接渲染 224×224 图像，并不是将官方 visual-cube 的 64×64 图像放大。**

准确说，LeWM 使用基于 OGBench 单方块环境修改的 `swm/OGBCube-v0`，另外采集图像数据：

- [采集脚本](https://github.com/galilai-group/stable-worldmodel/blob/main/scripts/data/collect_cube.py) 设置 `width=224, height=224`。
- 本地训练数据的 `pixels` 已经是 `(224,224,3)`。
- [当前推理配置](/home/coder/P-Rep/le-wm/config/eval/cube.yaml:6) 也设置为直接渲染 224×224。

我前面说“适合 visual”，指的是**模型使用视觉观测**，不意味着它直接使用官方 `visual-cube-*-v0` 的数据集。

**原版 LeWM 和当前 Cube 模型都没有把关节、夹爪、物体位置等状态向量作为模型输入。**

| 数据 | 是否输入模型 |
|---|---|
| 当前及历史图像 | 是，进入视觉编码器 |
| 动作数值 | 是，进入动作编码器，供动力学预测使用 |
| 目标图像 | 是，用于计算规划代价 |
| 关节、夹爪、物体位置、按钮状态等数值状态 | 否 |

本地 [encode 实现](/home/coder/P-Rep/le-wm/jepa.py:29) 只读取 `pixels` 和 `action`。虽然[训练配置](/home/coder/P-Rep/le-wm/config/train/data/ogb.yaml:5) 也加载了 `observation`，但模型没有使用它。

数值状态仍会用于**恢复模拟器状态、设置目标和判定成功**；这与把状态向量输入模型是两回事。因此它是 **图像观测＋动作条件的世界模型**。[论文说明](https://arxiv.org/html/2603.19312v1)


### OGBench visual Cube 原生 224×224 八卡评估（2026-10-05）

<!-- ogbench-evaluation: ogbench_visual_cube_lewm_native224_8gpu_20261005 -->

评估完成时间：`2026-10-05T18:52:23.806110-07:00`（America/Los_Angeles）。使用同一个固定 [cube-lewm checkpoint](/home/coder/P-Rep/le-wm/.stable-wm/checkpoints/cube-lewm)，未重新训练或微调。模型的观测和目标均由 MuJoCo 直接渲染为 224×224，没有经过 64×64 下采样。单方块结果包含从自采集图像到 visual 外观的迁移；double、triple、quadruple 均为**同一 single-cube checkpoint 的跨环境迁移**。

#### 与 PDF Table 5 对照

下表六个 baseline 列逐项取自 [OGBench PDF 第 21 页 Table 5](/home/coder/P-Rep/ogbench/ogbench.pdf)。全部数值为成功率百分比。**论文使用原生 64×64 观测；LeWM 按用户要求使用原生 224×224，因此观测分辨率不同。**任务、动作、目标和步数预算沿用官方设置。

| Table 5 数据集（baseline 64×64） | GCBC | GCIVL | GCIQL | QRL | CRL | HIQL | LeWM 原生224固定 checkpoint（均值 ± 评估种子标准差） | LeWM 评估性质 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| visual-cube-single-play-v0 | 5 ± 1 | 60 ± 5 | 30 ± 5 | 41 ± 15 | 31 ± 15 | 89 ± 0 | 0.00 ± 0.00 | 单方块 visual 外观迁移 |
| visual-cube-double-play-v0 | 1 ± 1 | 10 ± 2 | 1 ± 1 | 5 ± 0 | 2 ± 1 | 39 ± 2 | 0.00 ± 0.00 | 同一 checkpoint 跨环境迁移 |
| visual-cube-triple-play-v0 | 15 ± 2 | 14 ± 2 | 15 ± 1 | 16 ± 1 | 17 ± 2 | 21 ± 0 | 0.00 ± 0.00 | 同一 checkpoint 跨环境迁移 |
| visual-cube-quadruple-play-v0 | 8 ± 1 | 0 ± 0 | 7 ± 1 | 5 ± 1 | 4 ± 1 | 14 ± 1 | 0.00 ± 0.00 | 同一 checkpoint 跨环境迁移 |
| visual-cube-single-noisy-v0 | 14 ± 3 | 75 ± 3 | 48 ± 3 | 10 ± 5 | 39 ± 30 | 99 ± 0 | 0.00 ± 0.00 | 单方块 visual 外观迁移 |
| visual-cube-double-noisy-v0 | 5 ± 1 | 17 ± 4 | 22 ± 2 | 6 ± 2 | 6 ± 3 | 59 ± 3 | 0.00 ± 0.00 | 同一 checkpoint 跨环境迁移 |
| visual-cube-triple-noisy-v0 | 16 ± 1 | 18 ± 1 | 12 ± 1 | 9 ± 4 | 16 ± 1 | 23 ± 2 | 0.00 ± 0.00 | 同一 checkpoint 跨环境迁移 |
| visual-cube-quadruple-noisy-v0 | 9 ± 0 | 0 ± 0 | 2 ± 2 | 0 ± 0 | 8 ± 2 | 12 ± 8 | 0.00 ± 0.00 | 同一 checkpoint 跨环境迁移 |

**对照口径：**`play/noisy` 指训练数据分布，二者共享同一个评估环境。LeWM 此次只使用原有 自采集单方块 expert 数据训练的 checkpoint，因此同一变体在 play/noisy 两行重复显示同一组 评估结果；没有在这两套官方数据上分别训练 LeWM。论文 baseline 的 ± 是 4 个独立训练种子之间的 标准差，每个训练种子的分数还平均最后 3 次评估 checkpoint；LeWM 的 ± 是一个固定 checkpoint 在 4 个评估种子上的标准差。这是迁移性能参考，训练条件和误差条含义不同。

#### 五个官方目标的实测结果

每个环境共 1,000 回合；每个 task 共 200 回合（4 个评估种子 × 每种子 50 回合）。

| 环境 | task 1 | task 2 | task 3 | task 4 | task 5 | 总成功数 / 回合数 | 平均成功率 |
|---|---:|---:|---:|---:|---:|---:|---:|
| visual-cube-single-v0 | 0.00% (0/200) | 0.00% (0/200) | 0.00% (0/200) | 0.00% (0/200) | 0.00% (0/200) | 0/1000 | 0.00% |
| visual-cube-double-v0 | 0.00% (0/200) | 0.00% (0/200) | 0.00% (0/200) | 0.00% (0/200) | 0.00% (0/200) | 0/1000 | 0.00% |
| visual-cube-triple-v0 | 0.00% (0/200) | 0.00% (0/200) | 0.00% (0/200) | 0.00% (0/200) | 0.00% (0/200) | 0/1000 | 0.00% |
| visual-cube-quadruple-v0 | 0.00% (0/200) | 0.00% (0/200) | 0.00% (0/200) | 0.00% (0/200) | 0.00% (0/200) | 0/1000 | 0.00% |

| 环境 | seed 0 | seed 1 | seed 2 | seed 3 |
|---|---:|---:|---:|---:|
| single | 0.00% | 0.00% | 0.00% | 0.00% |
| double | 0.00% | 0.00% | 0.00% | 0.00% |
| triple | 0.00% | 0.00% | 0.00% | 0.00% |
| quadruple | 0.00% | 0.00% | 0.00% | 0.00% |

#### 评估设置与验证

| 项目 | 设置 |
|---|---|
| 环境与目标 | 官方 `visual-cube-{single,double,triple,quadruple}-v0`；`reset(options={task_id: 1..5})` 返回的官方目标图像 |
| 单回合预算 | single 200；double 500；triple 1,000；quadruple 1,000 个环境 step，与注册环境一致 |
| 成功判定 | 官方最终 `info[success]`；成功终止或达到 TimeLimit 后停止；对 5 个 task 等权平均 |
| 图像输入 | visual 相机和半透明机械臂，渲染器 width=height=224；观测与目标均为原生 224×224 RGB；ImageNet normalization + Resize224（尺寸不变） |
| 数值输入 | 仅动作用于动作编码；没有把关节、夹爪、物体位置等 state 向量送入模型 |
| 规划 | CEM：300 candidates、30 iterations、30 elites、variance_scale=1；horizon=5、action_block=5、receding_horizon=5（每 25 个环境 step 重规划） |
| 历史 | 延续当前推理入口的 1 帧观测历史；预测器在想象 rollout 中保留至多 3 帧上下文 |
| 动作处理 | 原训练 HDF5 有限动作行拟合 StandardScaler（2,000,000 行）；执行动作 clip 到 [-1,1] |
| 并行 | 8 张 H200；每 GPU 1 个规划进程，每批 10 个独立模拟器；80 个任务批次动态分配 |
| 数值精度 | float32；PyTorch matmul precision `highest`；predictor compile=False |
| 终止处理 | AsyncVectorEnv 使用 spawn；关闭自动 reset；完成的环境保持终止状态，不计入新回合 |

模型 encoder 的 forward 前置检查在 8 张 GPU 上直接读取实际输入张量，均为 `[batch, 3, 224, 224]`、float32。已规划动作执行期间跳过重复的无状态图像预处理；在 1、2、10 个环境以及混合终止状态下，25 步缓存动作与原实现逐值相同。

固定了模拟器、NumPy 与 CEM 的随机种子。官方 Cube 在生成目标图像时还会调用 action_space.sample() 做两步稳定动作，而 action_space 属性每次创建新的 Box；此次保留官方采样行为，因此固定 reset seed 不保证目标图像逐像素重复。

修复了原 LeWM 目标 embedding 在多环境批量规划时缺少 candidate 轴的问题，仅在评估适配器中 增加广播维度，未修改权重或代价函数。8 个 GPU 均使用真实起始图像，验证了批量代价与原来逐环境、matmul precision=highest 的代价一致（rtol=atol=1e-3），且候选动作最优索引一致。

Checkpoint weights SHA-256：`2839a907362f403f9136383016e91774373a295d958ae75121791f22a9fddf89`。

运行版本：torch=2.13.0; torchvision=0.28.0; stable-worldmodel=0.1.1; ogbench=1.2.1; mujoco=3.11.0。

官方环境的正常推理和统计流程通过验证；成功率表示这个固定模型的任务性能。训练数据来自原版单方块 224×224 渲染，visual 测试改变了相机、颜色和透明机械臂外观，多方块同时改变物体数量。本次已消除 64→224 放大，但仍存在训练与测试的外观和任务差异；这些结果不能代表重新训练后的 LeWM 上限。

#### 可复现记录

- [完整指标与逐回合结果](/home/coder/P-Rep/le-wm/outputs/ogbench_visual_cube_lewm_native224_8gpu_20261005/metrics.json)；[运行配置](/home/coder/P-Rep/le-wm/outputs/ogbench_visual_cube_lewm_native224_8gpu_20261005/config.json)；[动作 normalizer](/home/coder/P-Rep/le-wm/outputs/ogbench_visual_cube_lewm_native224_8gpu_20261005/action-normalizer.json)。
- [Table 5 baseline 原值](/home/coder/P-Rep/le-wm/outputs/ogbench_visual_cube_lewm_native224_8gpu_20261005/ogbench_table5_visual_cube_baselines.json)；[本次执行脚本快照](/home/coder/P-Rep/le-wm/outputs/ogbench_visual_cube_lewm_native224_8gpu_20261005/eval_ogbench_visual_cube.py)。
- [80 个 task/seed 原始记录](/home/coder/P-Rep/le-wm/outputs/ogbench_visual_cube_lewm_native224_8gpu_20261005/jobs)；[20 个目标的推理视频与起始图](/home/coder/P-Rep/le-wm/outputs/ogbench_visual_cube_lewm_native224_8gpu_20261005/videos)。视频为 MuJoCo 原生渲染的 224×224 当前观测，无白边、拼接或叠字。模型收到的也是这一原生 224×224 图像；20 个视频回合已包含在 4,000 回合统计中。
- [视频尺寸及模型输入检查](/home/coder/P-Rep/le-wm/outputs/ogbench_visual_cube_lewm_native224_8gpu_20261005/video-manifest.json)。

复现命令：

```bash
cd /home/coder/P-Rep/le-wm
.venv/bin/python eval_ogbench_visual_cube.py \
  --gpus 0 1 2 3 4 5 6 7 --seeds 0 1 2 3 \
  --variants single double triple quadruple --tasks 1 2 3 4 5 \
  --episodes-per-task 50 --parallel-envs 10 --matmul-precision highest \
  --observation-size 224 --video-size 224
```

<!-- lewm-finetune:lewm_visual_cube_ft8_20261005_204432:start -->
### OGBench visual Cube 微调旧流程（lewm_visual_cube_ft8_20261005_204432，已停止）

用户要求训练也采用渲染器原生224。此批使用64放大图，已停止；后续原生224流程从原始checkpoint重新初始化，不续训此批权重。

官方主 NPZ 用于训练，独立 `-val.npz` 仅用于验证及选择 checkpoint。在线成功率在模拟器的 5 个官方目标任务上测量；`-val` 不是在线测试集。

八个模型分别从原始 `cube-lewm` checkpoint 严格加载全部权重，然后在四种 Cube × play/noisy 的各自 train 数据上独立微调。原始权重 SHA256：`2839a907362f403f9136383016e91774373a295d958ae75121791f22a9fddf89`。

每模型 80,000 次更新，batch size=128，AdamW lr=1e-05，weight decay=0.001，5% warmup（起始 lr=目标值的 10%）后 cosine，bf16 mixed，梯度裁剪 1.0。history=3，prediction=1，frameskip=5，预测 MSE + 0.09×SIGReg。所有模型参数参与更新。

八张卡各训练一个模型，统一更新次数以平衡不同数据量；每遍 train 数据后和最后一步均在完整官方 val 数据上验证，按最低 val loss 选 best。动作标准化只拟合各自 train 数据的非 terminal 动作，并用于该模型的训练、验证和在线规划。qpos/qvel 等数值状态不输入模型。

离线训练图像原生 64×64，按照 LeWM 的 ImageNet normalize + bilinear antialias resize 输入 224×224；在线评测输入和原始视频从渲染器直接取得 224×224，无白边。训练/评测图像采样分辨率不同；PDF baseline 使用标准 64 像素协议，因此本表应作为参照比较。

训练完成后自动启动八卡共用任务队列：每模型 5 个 task × 4 个 evaluation seed（0,1,2,3）× 50 episodes = 1,000 episodes，总计 8,000。CEM 与之前固定 checkpoint 的测试完全一致：samples=300，iterations=30，topk=30，horizon=5，receding=5，action block=5，observed history=1。

下表 baseline 严格来自本地 ogbench.pdf 第 21 页 Table 5。baseline 的标准差来自 4 个独立训练种子；本次每份数据只训练 1 个模型，LeWM 微调结果的标准差来自 4 个评测种子，统计含义不同。先前 LeWM 固定权重在四种 Cube 上均为 0/1,000；该旧结果对 play/noisy 共用，微调后每行是各自数据训练的独立模型。

| Dataset | GCBC | GCIVL | GCIQL | QRL | CRL | HIQL | LeWM 原 checkpoint | LeWM 微调 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| visual-cube-single-play-v0 | 5 ± 1 | 60 ± 5 | 30 ± 5 | 41 ± 15 | 31 ± 15 | 89 ± 0 | 0 ± 0 | 已停止：64→224 图像 |
| visual-cube-double-play-v0 | 1 ± 1 | 10 ± 2 | 1 ± 1 | 5 ± 0 | 2 ± 1 | 39 ± 2 | 0 ± 0 | 已停止：64→224 图像 |
| visual-cube-triple-play-v0 | 15 ± 2 | 14 ± 2 | 15 ± 1 | 16 ± 1 | 17 ± 2 | 21 ± 0 | 0 ± 0 | 已停止：64→224 图像 |
| visual-cube-quadruple-play-v0 | 8 ± 1 | 0 ± 0 | 7 ± 1 | 5 ± 1 | 4 ± 1 | 14 ± 1 | 0 ± 0 | 已停止：64→224 图像 |
| visual-cube-single-noisy-v0 | 14 ± 3 | 75 ± 3 | 48 ± 3 | 10 ± 5 | 39 ± 30 | 99 ± 0 | 0 ± 0 | 已停止：64→224 图像 |
| visual-cube-double-noisy-v0 | 5 ± 1 | 17 ± 4 | 22 ± 2 | 6 ± 2 | 6 ± 3 | 59 ± 3 | 0 ± 0 | 已停止：64→224 图像 |
| visual-cube-triple-noisy-v0 | 16 ± 1 | 18 ± 1 | 12 ± 1 | 9 ± 4 | 16 ± 1 | 23 ± 2 | 0 ± 0 | 已停止：64→224 图像 |
| visual-cube-quadruple-noisy-v0 | 9 ± 0 | 0 ± 0 | 2 ± 2 | 0 ± 0 | 8 ± 2 | 12 ± 8 | 0 ± 0 | 已停止：64→224 图像 |

运行目录：`/home/coder/P-Rep/le-wm/outputs/lewm_visual_cube_ft8_20261005_204432`。tmux session：`lewm-ogbench-ft8`。

脚本：`launch_ogbench_finetune_8gpu.sh` → `run_ogbench_finetune_pipeline.py` → `finetune_ogbench_visual_cube.py` / `eval_ogbench_visual_cube.py`。

[W&B 项目](https://forge.coreweave.com/wandb/visual-intelligence-laboratory/P-Rep)，各模型的 loss 链接保存在对应 training 子目录的 `wandb-run.json`。密钥存储在仓库外的私有环境文件中。
<!-- lewm-finetune:lewm_visual_cube_ft8_20261005_204432:end -->

<!-- lewm-finetune:lewm_visual_cube_native224_ft8_20261005_211637:start -->
### OGBench visual Cube 分数据集微调与自动评测（lewm_visual_cube_native224_ft8_20261005_211637）

官方主 NPZ 用于训练，独立 `-val.npz` 仅用于验证及选择 checkpoint。在线成功率在模拟器的 5 个官方目标任务上测量；`-val` 不是在线测试集。

八个模型分别从原始 `cube-lewm` checkpoint 严格加载全部权重，然后在四种 Cube × play/noisy 的各自 train 数据上独立微调。原始权重 SHA256：`2839a907362f403f9136383016e91774373a295d958ae75121791f22a9fddf89`。

每模型 80,000 次更新，batch size=128，AdamW lr=1e-05，weight decay=0.001，5% warmup（起始 lr=目标值的 10%）后 cosine，bf16 mixed，梯度裁剪 1.0。history=3，prediction=1，frameskip=5，预测 MSE + 0.09×SIGReg。所有模型参数参与更新。

八张卡各训练一个模型，统一更新次数以平衡不同数据量；每遍 train 数据后和最后一步均在完整官方 val 数据上验证，按完整独立val集最低预测MSE选best；连续轨迹组批下的SIGReg组合指标保留作诊断。动作标准化只拟合各自 train 数据的非 terminal 动作，并用于该模型的训练、验证和在线规划。qpos/qvel 等数值状态不输入模型。

训练及验证图像由官方 NPZ 保存的 qpos/qvel 还原同一场景，直接用 MuJoCo 渲染器生成 224×224 RGB；保留官方透明机械臂、front_pixels 相机及静态物体配色。图片用无损 Zstd 存储，训练仅做 ImageNet normalize，禁止 64 放大。qpos/qvel 只用于离线重渲染，不作为 LeWM 输入。重渲染时不执行动作或额外模拟 step，原始动作、状态次序和 train/val 划分保持一致。每个数据 shard 在 64 分辨率上与原官方图片核对平均像素误差 <1/255 后生成原生224缓存。

在线评测输入、goal 图像和无白边原始视频也均由渲染器直接产生 224×224。本实验训练/验证/测试分辨率统一为原生224；PDF baseline 使用标准64像素协议，本表应作为参照比较。

训练完成后自动启动八卡共用任务队列：每模型 5 个 task × 4 个 evaluation seed（0,1,2,3）× 50 episodes = 1,000 episodes，总计 8,000。CEM 与之前固定 checkpoint 的测试完全一致：samples=300，iterations=30，topk=30，horizon=5，receding=5，action block=5，observed history=1。

下表 baseline 严格来自本地 ogbench.pdf 第 21 页 Table 5。baseline 的标准差来自 4 个独立训练种子；本次每份数据只训练 1 个模型，LeWM 微调结果的标准差来自 4 个评测种子，统计含义不同。先前 LeWM 固定权重在四种 Cube 上均为 0/1,000；该旧结果对 play/noisy 共用，微调后每行是各自数据训练的独立模型。

| Dataset | GCBC | GCIVL | GCIQL | QRL | CRL | HIQL | LeWM 原 checkpoint | LeWM 微调 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| visual-cube-single-play-v0 | 5 ± 1 | 60 ± 5 | 30 ± 5 | 41 ± 15 | 31 ± 15 | 89 ± 0 | 0 ± 0 | 0.00 ± 0.00 |
| visual-cube-double-play-v0 | 1 ± 1 | 10 ± 2 | 1 ± 1 | 5 ± 0 | 2 ± 1 | 39 ± 2 | 0 ± 0 | 0.00 ± 0.00 |
| visual-cube-triple-play-v0 | 15 ± 2 | 14 ± 2 | 15 ± 1 | 16 ± 1 | 17 ± 2 | 21 ± 0 | 0 ± 0 | 0.10 ± 0.17 |
| visual-cube-quadruple-play-v0 | 8 ± 1 | 0 ± 0 | 7 ± 1 | 5 ± 1 | 4 ± 1 | 14 ± 1 | 0 ± 0 | 0.00 ± 0.00 |
| visual-cube-single-noisy-v0 | 14 ± 3 | 75 ± 3 | 48 ± 3 | 10 ± 5 | 39 ± 30 | 99 ± 0 | 0 ± 0 | 0.00 ± 0.00 |
| visual-cube-double-noisy-v0 | 5 ± 1 | 17 ± 4 | 22 ± 2 | 6 ± 2 | 6 ± 3 | 59 ± 3 | 0 ± 0 | 0.10 ± 0.17 |
| visual-cube-triple-noisy-v0 | 16 ± 1 | 18 ± 1 | 12 ± 1 | 9 ± 4 | 16 ± 1 | 23 ± 2 | 0 ± 0 | 1.40 ± 0.66 |
| visual-cube-quadruple-noisy-v0 | 9 ± 0 | 0 ± 0 | 2 ± 2 | 0 ± 0 | 8 ± 2 | 12 ± 8 | 0 ± 0 | 0.00 ± 0.00 |

运行目录：`/home/coder/P-Rep/le-wm/outputs/lewm_visual_cube_native224_ft8_20261005_211637`。tmux session：`lewm-ogbench-ft8`。

脚本：`launch_ogbench_finetune_8gpu.sh` → `run_ogbench_finetune_pipeline.py` → `prepare_ogbench_native224.py`（八卡共享 shard 队列重渲染）→ `finetune_ogbench_visual_cube.py`（八模型各一张卡微调）→ `eval_ogbench_visual_cube.py`（八卡共用 task 队列自动评测）。

[W&B 项目](https://forge.coreweave.com/wandb/visual-intelligence-laboratory/P-Rep)，各模型的 loss 链接保存在对应 training 子目录的 `wandb-run.json`。密钥存储在仓库外的私有环境文件中。

微调和全部在线评测已完成，8个模型各80,000步，合计成功16/8000（0.20%）；本次配置的成功率仍很低。checkpoint、逐episode记录、40段原生视频及输入检查均保存在运行目录。结果与PDF baseline也已写入 [README.md](README.md#ogbench-visual-cube-原生224微调与评测本地实验)。

| Dataset | 成功数 / episodes | best epoch | best val prediction MSE |
|---|---:|---:|---:|
| visual-cube-single-play-v0 | 0 / 1000 | 11 | 0.003454 |
| visual-cube-single-noisy-v0 | 0 / 1000 | 9 | 0.012328 |
| visual-cube-double-play-v0 | 0 / 1000 | 9 | 0.005578 |
| visual-cube-double-noisy-v0 | 1 / 1000 | 10 | 0.013212 |
| visual-cube-triple-play-v0 | 1 / 1000 | 3 | 0.005754 |
| visual-cube-triple-noisy-v0 | 14 / 1000 | 3 | 0.011127 |
| visual-cube-quadruple-play-v0 | 0 / 1000 | 2 | 0.006280 |
| visual-cube-quadruple-noisy-v0 | 0 / 1000 | 3 | 0.010974 |

| Dataset | Task | Success % |
|---|---|---:|
| visual-cube-single-play-v0 | 1: task1_horizontal | 0.00 |
| visual-cube-single-play-v0 | 2: task2_vertical1 | 0.00 |
| visual-cube-single-play-v0 | 3: task3_vertical2 | 0.00 |
| visual-cube-single-play-v0 | 4: task4_diagonal1 | 0.00 |
| visual-cube-single-play-v0 | 5: task5_diagonal2 | 0.00 |
| visual-cube-single-noisy-v0 | 1: task1_horizontal | 0.00 |
| visual-cube-single-noisy-v0 | 2: task2_vertical1 | 0.00 |
| visual-cube-single-noisy-v0 | 3: task3_vertical2 | 0.00 |
| visual-cube-single-noisy-v0 | 4: task4_diagonal1 | 0.00 |
| visual-cube-single-noisy-v0 | 5: task5_diagonal2 | 0.00 |
| visual-cube-double-play-v0 | 1: task1_single_pnp | 0.00 |
| visual-cube-double-play-v0 | 2: task2_double_pnp1 | 0.00 |
| visual-cube-double-play-v0 | 3: task3_double_pnp2 | 0.00 |
| visual-cube-double-play-v0 | 4: task4_swap | 0.00 |
| visual-cube-double-play-v0 | 5: task5_stack | 0.00 |
| visual-cube-double-noisy-v0 | 1: task1_single_pnp | 0.50 |
| visual-cube-double-noisy-v0 | 2: task2_double_pnp1 | 0.00 |
| visual-cube-double-noisy-v0 | 3: task3_double_pnp2 | 0.00 |
| visual-cube-double-noisy-v0 | 4: task4_swap | 0.00 |
| visual-cube-double-noisy-v0 | 5: task5_stack | 0.00 |
| visual-cube-triple-play-v0 | 1: task1_single_pnp | 0.50 |
| visual-cube-triple-play-v0 | 2: task2_triple_pnp | 0.00 |
| visual-cube-triple-play-v0 | 3: task3_pnp_from_stack | 0.00 |
| visual-cube-triple-play-v0 | 4: task4_cycle | 0.00 |
| visual-cube-triple-play-v0 | 5: task5_stack | 0.00 |
| visual-cube-triple-noisy-v0 | 1: task1_single_pnp | 7.00 |
| visual-cube-triple-noisy-v0 | 2: task2_triple_pnp | 0.00 |
| visual-cube-triple-noisy-v0 | 3: task3_pnp_from_stack | 0.00 |
| visual-cube-triple-noisy-v0 | 4: task4_cycle | 0.00 |
| visual-cube-triple-noisy-v0 | 5: task5_stack | 0.00 |
| visual-cube-quadruple-play-v0 | 1: task1_double_pnp | 0.00 |
| visual-cube-quadruple-play-v0 | 2: task2_quadruple_pnp | 0.00 |
| visual-cube-quadruple-play-v0 | 3: task3_pnp_from_square | 0.00 |
| visual-cube-quadruple-play-v0 | 4: task4_cycle | 0.00 |
| visual-cube-quadruple-play-v0 | 5: task5_stack | 0.00 |
| visual-cube-quadruple-noisy-v0 | 1: task1_double_pnp | 0.00 |
| visual-cube-quadruple-noisy-v0 | 2: task2_quadruple_pnp | 0.00 |
| visual-cube-quadruple-noisy-v0 | 3: task3_pnp_from_square | 0.00 |
| visual-cube-quadruple-noisy-v0 | 4: task4_cycle | 0.00 |
| visual-cube-quadruple-noisy-v0 | 5: task5_stack | 0.00 |
<!-- lewm-finetune:lewm_visual_cube_native224_ft8_20261005_211637:end -->

### 原生224图片替换本地 OGBench 数据

按用户要求，当前后台流程在全部原生224重渲染及校验完成后，自动把 `/home/coder/P-Rep/ogbench` 中对应的 16 份 visual-cube NPZ（四种 Cube × play/noisy × train/val）里的 `observations` 原位替换成原生224 RGB，再删除对应 `.lewm-npy-cache` 中旧的64图片。文件名和标准 NPZ读取方式保留，不留64图片备份；所有非图片成员逐字节 SHA256核对，动作、qpos/qvel、terminal、帧序和 train/val 划分保持一致。训练继续读取同一批原生224图片的无损 HDF5缓存，随后自动八卡训练与评测。

此步骤已由本次 `lewm-ogbench-ft8` 后台流程执行：训练入口在模型初始化前检查本次运行的 `native224-replacement-request.json`，并完成本数据集的 train/val替换。16份正式NPZ均已完成原生224替换及旧64缓存清理；完成记录见下述manifest。实现位于 `replace_ogbench_native224.py`，全部完成后的记录为 `/home/coder/P-Rep/ogbench/native224-replacement-manifest.json`；官方原始下载校验值不再适用于替换后的本地文件。

替换流程已用独立的16份小样本数据并行验证，检查了完整 ZIP CRC、全部224像素逐像素一致、非图片数组完整保留、旧64缓存删除、重复调用不重写文件，以及替换后的 LeWM训练读取器。结果：`outputs/native224_npz_replacement_check/result.json`，全部通过；此验证未修改正式数据。

### 官方64与原生224的同帧对齐核对

对16份 visual-cube train/val数据抽查96组同帧64/224图片，场景、front_pixels相机、机械臂透明度、物体及机械臂姿态均保持一致。224来自同一行qpos/qvel的原生渲染，不执行额外模拟step；动作、帧序和轨迹划分保留。四种Cube的同帧对比见 `outputs/ogbench_native224_resolution_check/64_vs_native224.png`。

截至本次核对，已完成shard的3,978次“保存状态→64重渲染→官方64”校验全部通过，平均像素MAE=0.213/255，最高单帧MAE=0.479/255。96组原生224缩小回64后的平均MAE=1.856/255，最高单帧MAE=2.067/255；高分辨率栅格化及抗锯齿不同，缩小后不会与官方64逐像素完全相同。所有原生224样本均不同于64的bilinear放大，图像清晰度提升来自原生渲染。详细核对记录：`outputs/ogbench_native224_resolution_check/metrics.json`。

### 微调loss核对与最终checkpoint选择修正

16份原生224 NPZ替换已全部完成，八个模型继续各自80,000步的训练。训练loss从起初约0.37–0.80下降至约0.08–0.10，预测MSE和梯度均为有限值，日志没有NaN/Inf或异常退出；已完成验证的模型，完整独立val集预测MSE约0.0039–0.0153。quadruple模型当时仍在第一遍训练，尚未完成第一次完整val。

发现验证组合loss约2–6主要由SIGReg约24–65构成，而训练SIGReg约0.8–1.0。验证reader按连续轨迹顺序组batch，相邻clips的强相关性导致SIGReg的批次分布统计与随机训练batch不一致。对single-play和double-play各取同一组1,024个验证样本、同一模型与相同embedding，只随机重组batch，SIGReg分别从35.13降为6.16、61.89降为10.62，降低约5.7–5.8倍；该实验验证了组批对统计量的影响，不能将原验证总loss直接与训练总loss作大小比较。完整记录为 `/home/coder/P-Rep/le-wm/outputs/lewm_visual_cube_native224_ft8_20261005_211637/validation-sigreg-audit/same_examples_regrouping.json`。

为避免原组合指标误导checkpoint选择，当前后台流程已注册 `validation-selection-request.json`：训练目标、更新次数和正在执行的训练保持原样，自动评测启动前从所有已保存epoch中按完整独立val集最低预测MSE选checkpoint。原组合指标、原best记录和全部训练权重保留。`select_ogbench_checkpoint.py` 在实际加载评测模型前应用选择，保存每模型 `prediction-checkpoint-selection.json`，并将评测manifest中的路径与SHA256更新为实际选中权重。报告中的best val loss应解释为完整val预测MSE。选择逻辑的并发、一致性、完成条件及权重保留测试均通过：`/home/coder/P-Rep/le-wm/outputs/validation_selection_check/result.json`。W&B中原 `val/loss` 组合曲线保留，检查预测效果应查看 `val/pred_loss`；在线评测已完成，成功率见上方微调结果表及README。

