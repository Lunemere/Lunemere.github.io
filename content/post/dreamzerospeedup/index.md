+++
date = '2026-11-02T06:37:22-08:00'
draft = true
title = 'DreamZero 在 Jetson Thor 上的推理加速剖析'
tags = ["DreamZero", "Thor"]
categories = ["推理加速"]
math = true
+++

## 一、性能剖析：时间都花在哪里

做优化之前，先要知道时间花在了哪里。优先在 Thor 上运行一遍 baseline，记录一个 denoise step 以及一个 transformer block 内各部分的耗时，以此找出性能热点，并方便后续比对。

### 1.1 测试环境与设置

| 项目 | 内容 |
| --- | --- |
| 硬件 | NVIDIA Jetson Thor（sm_110，128 GB 统一内存） |
| 软件 | 容器 `lightx2v-thor-polish`，CUDA 13.0，BF16 推理 |
| 模型结构 | DiT 40 层，hidden 5120，40 头 × 128，FFN 13824 |
| 序列长度 | 每个 CFG 分支 1785 个 token（2 帧 × 880 视频 token + 24 action + 1 state） |
| 推理设置 | 共 10 个 chunk；每个 chunk 16 个 scheduler step，其中只有 8 步真正调用 DiT（`dit_step_mask`），其余步复用上一步的预测；开启 CFG，cond / uncond 两个分支串行执行 |

这里的 **一个 denoise step** 指一次真正调用 DiT 的去噪步：pre-infer + cond、uncond 两个分支各跑一遍 40 层 transformer block + video head / action decoder + CFG 合成 + scheduler 更新。

### 1.2 测量方法

- **配置**：原始版本的 self-attention 和 cross-attention 都使用 FlashAttention 2（FA2）。GPU 频率上限为默认的 1575 MHz。
- **端到端耗时**：直接运行推理脚本，读取 LightX2V 自带的 profiling 日志（`Load models`、`Run DiT`、每个 chunk 和每个 step 的耗时）。
- **分项耗时**：用一个包装脚本启动推理，**不修改仓库代码**。在第 6 个 chunk 中连续抓取第 2～4 个 DiT step 的 GPU trace，对 3 个 step 取平均。选第 6 个 chunk 是因为此时推理已进入稳定状态（见 1.3 节）。
- 抓取时**只记录 GPU kernel，不记录 CPU 端事件**。在 Thor 上同时记录 CPU 端事件会明显拖慢推理（实测 step 变慢约 30%），只记录 GPU kernel 时 step 只慢约 3%（1967 ms，不抓取时同一位置约 1912 ms）。
- 每个 kernel 属于哪一部分，通过与一次参考运行对照确定：参考运行中给各个函数、各个权重模块都打了标签，可以知道一个 block 内每个 kernel 属于哪一部分；而每个 block 执行的 kernel 序列是固定的，因此可以逐个对应（3 个 step 共 240 次 block 执行，kernel 序列全部与参考一致）。
- 统计的是 **GPU kernel 时间**。一个 step 中 GPU 的忙碌时间约占 99%（1956 ms 忙碌 / 1967 ms 总耗时），所以 kernel 时间之和基本等于 step 的实际耗时，CPU 下发开销不是瓶颈。

### 1.3 端到端耗时

| 阶段 | 耗时 |
| --- | ---: |
| 模型加载 | 63.4 s |
| Run DiT（10 个 chunk） | 212.9 s |
| 总耗时 | 277.6 s |
| 单个 denoise step（chunk 5～10 稳态均值） | 1.926 s |
| 单个 denoise step（全部 80 步均值） | 2.113 s |

各 chunk 的耗时（每个 chunk 含 8 个 denoise step；chunk 耗时还包括 DiT step 之外的 KV cache 更新等）：

| chunk | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| chunk 耗时 (s) | 22.5 | 22.2 | 23.1 | 23.3 | 17.7 | 17.8 | 18.9 | 19.9 | 17.6 | 17.7 |
| denoise step 均值 (s) | 2.184 | 2.380 | 2.496 | 2.510 | 1.780 | 1.910 | 2.029 | 2.151 | 1.782 | 1.903 |

补充说明：

- 同一个 segment 内，KV cache 随 chunk 增长，self-attention 要处理的 key/value 越来越多，step 耗时也从 1.78 s 逐步上升到 2.15 s（chunk 5～8）；进入新的 segment 时 cache 清空，耗时回落。
- 第一个 segment（chunk 1～4）明显偏慢（2.2～2.5 s/step），比后面 segment 中相同位置的 chunk 每个 step 慢约 0.4～0.5 s。

### 1.4 一个 denoise step 的耗时拆分

第 6 个 chunk，单位 ms（GPU kernel 时间，3 个 step 平均）：

| 部分 | 耗时 | 占比 |
| --- | ---: | ---: |
| DiT blocks（40 层 × 2 个 CFG 分支） | 1951.4 | 99.8% |
| 其余部分（pre-infer、video head、action decoder、CFG 合成、scheduler 更新） | 4.7 | 0.2% |
| **合计** | **1956.1** | |
| 实测 step 墙钟时间（含 profiler 开销） | 1967 | |

结论很直接：**99% 以上的时间都在 40 层 transformer block 里**，pre-infer、head、scheduler 加起来不到 1%。优化必须集中在 block 内部。

### 1.5 一个 transformer block 的耗时拆分

一个 block 跑一个 CFG 分支的耗时，单位 µs（= 一个 step 中该部分的总时间 ÷ 80）：

| 排名 | 部分 | 耗时 | 占比 | 主要 kernel |
| --- | --- | ---: | ---: | --- |
| 1 | self-attention | 3203.7 | 13.1% | FA2 |
| 2 | q/k RMSNorm + RoPE | 3081.3 | 12.6% | 逐元素运算 + flashinfer RoPE |
| 3 | QKV 投影 | 2985.5 | 12.2% | 3 次 cuBLAS GEMM |
| 4 | FFN up + GELU | 2957.0 | 12.1% | cuBLAS GEMM + GELU |
| 5 | FFN down | 2936.5 | 12.0% | cuBLAS GEMM |
| 6 | cross-attn Q 投影 + RMSNorm | 2367.2 | 9.7% | cuBLAS GEMM + 逐元素运算 |
| 7 | AdaLN 调制参数（`table + embed0`） | 1427.4 | 5.9% | 逐元素加法 |
| 8 | cross-attn O 投影 | 979.0 | 4.0% | cuBLAS GEMM |
| 9 | self-attn O 投影 | 950.8 | 3.9% | cuBLAS GEMM |
| 10 | cross-attention（text + image） | 857.1 | 3.5% | 2 次 FA2 + 加法 |
| 11 | KV cache 拼接 / 写回 | 729.0 | 3.0% | DtoD memcpy |
| 12 | 残差 + norm2 + modulate | 538.7 | 2.2% | 加法 + LayerNorm + `fuse_scale_shift` |
| 13 | gate 残差 + norm3 | 525.3 | 2.2% | 乘、加 + LayerNorm |
| 14 | norm1 + modulate | 435.3 | 1.8% | LayerNorm + `fuse_scale_shift` |
| 15 | FFN gate 残差 | 418.5 | 1.7% | 乘、加 |
| | **合计** | **24392.5** | | |

> 说明：cross-attention 的 K/V 投影只在第一次使用时计算并缓存，稳态 step 中不出现，因此不在表中。

### 1.6 热点分析

热点可以归纳为三类：

1. **访存型小算子占了 37%，仅次于 GEMM**：这类算子计算量很小，本不该这么慢，是最值得先动手的地方：
   - **RMSNorm**：每层有 3 次（self-attn 的 q、k 和 cross-attn 的 q），每次被拆成 fp32 类型转换、平方、求均值、乘法、转回 BF16 等 9 个 kernel（其中 6 个要完整读写整个张量），每次约 1.39 ms。加上 RoPE（约 0.31 ms），「q/k RMSNorm + RoPE」与「cross-attn Q 投影 + RMSNorm」中的 norm 部分每层共约 4.5 ms，一个 step 约 358 ms。
   - **AdaLN 调制参数**：`table + embed0` 被展开成逐 token 的 `(seq, 6, 5120)` 大张量，但其中实际不同的行只有几行。每层 1.43 ms，一个 step 约 114 ms。
   - **KV cache 拼接**：每层先把历史 K/V 和当前 K/V 拷贝到临时缓冲区，再送进 attention。每层约 0.73 ms 的 DtoD memcpy，一个 step 约 58 ms。
   - 其余是 GELU、gate 乘加、残差等未融合的逐元素运算，以及 LayerNorm 与 modulate 分开执行带来的额外读写。
2. **GEMM 占 47%**：QKV（3 次独立的 GEMM）、self-attn O、cross-attn Q / O、FFN up / down 共 8 个 GEMM，全部走 cuBLAS（nvjet kernel）。其中 FFN up / down 是单次耗时最大的两个 GEMM（各约 2.6～2.9 ms）。
3. **Attention 占 16%**：FA2 self-attention 每层约 3.2 ms，是单项耗时最大的算子；两个 cross-attention 合计约 0.69 ms。FA2 是为 Ampere 架构设计的，在 Thor 上用不到 Blackwell 的新硬件特性，self-attention 有明显的提升空间。

因此优化的优先级是：先消除访存型小算子的开销（融合 norm / modulate / 残差、避免 AdaLN 参数展开和 KV 拷贝），再提升 GEMM 的效率，并把 self-attention 换成为 Blackwell 设计的实现。

## 二、优化访存型小算子

> **背景：Thor 的内存带宽很低**：Jetson Thor 使用 LPDDR5X 统一内存，CPU 和 GPU 共享约 273 GB/s 的带宽，只有 H100（约 3.35 TB/s）的十二分之一左右。在数据中心 GPU 上耗时可以忽略的逐元素运算，在 Thor 上会变得非常显眼。

第一节的剖析显示，原始版本中 block 内的访存型小算子每个 denoise step 约耗时 728 ms，仅次于 GEMM（912 ms）。这类算子的计算量很小，耗时几乎全部花在读写显存上。每层的 hidden state 有 1785 × 5120 个 BF16 元素（约 18 MB），每多一个逐元素 kernel，就要把这么大的数据（有时还是 fp32 版本，大小翻倍）从内存读一遍、再写一遍。

因此优化的思路很明确：**把多个小算子合并成一个 kernel，避免生成不必要的中间张量，让数据尽量只读写一次**。下面逐一介绍。

### 2.1 AdaLN 调制参数：只保存不同的行

DiT 的每一层都用 AdaLN 根据时间步（timestep）调节 LayerNorm 的输出：每个 token 需要 6 组向量（两组 shift、两组 scale、两组 gate，每组 5120 维），由该层的一个可学习参数表加上这个 token 的 time embedding 得到。

原始实现先为每个 token 都算一份 time embedding，再在每一层把它和参数表相加，为全部 1785 个 token 生成 6 × 5120 的调制参数。这个张量约 110 MB，每层都要完整地读一遍、写一遍，仅这一次加法就耗时 1.43 ms。

但 DreamZero 中同一类 token 共用一个时间步：所有视频 token 一个，所有 action token 一个，state token 一个。所以 1785 行调制参数里，真正不同的只有大约 3 行，其余都是重复的。因此可以这样处理：

- 在 pre-infer 阶段，把时间步相同的连续 token 归为一"段"，每段只保留一行 time embedding，另外记录每个 token 属于哪一段（一个很小的整数索引）。time embedding 的 MLP 也只对这几行计算，不再对全部 1785 行计算。它的最后一层原来是一个 1785 × 30720 × 5120 的大 GEMM，现在只需算几行；time embedding 按 chunk 的起始帧位置和 step 编号缓存，第一个 segment（chunk 1～4）中每个 step 因此从约 7 ms 降到约 2 ms，之后的 chunk 直接复用缓存。
- 每层把参数表加到这几行上即可，耗时从 1.43 ms 降到 3 µs。
- 后面所有用到调制参数的运算（modulate、gate 残差），都改为在 kernel 内部根据 token 的段索引去取对应的那一行，再也不生成逐 token 的大张量。

### 2.2 RMSNorm 融合，顺带融合 RoPE

每层要对 attention 的 query、key 做 3 次 RMSNorm（self-attention 的 q、k 和 cross-attention 的 q）。原始实现直接用 PyTorch 的基本运算拼出 RMSNorm：转成 fp32、求平方、求均值、开方取倒数、相乘、转回 BF16、乘以权重。每一步都是一个独立的 kernel，中间结果都是 fp32 的完整张量，每次 RMSNorm 约 1.4 ms。self-attention 的 q、k 做完 RMSNorm 后，还要再跑一个单独的 RoPE（旋转位置编码）kernel。

现在用一个 Triton kernel 完成整个 RMSNorm：

- 每个线程块负责几行。先读一遍这几行，算出平方和；再读一遍做归一化、乘以权重，然后写出结果。第二次读取时数据还在 L2 cache 中，基本不再访问内存。这样整个 RMSNorm 只需从内存读一次、写一次。
- 累加平方和时，严格按照 PyTorch 原实现的分组和累加顺序进行，保证浮点舍入完全相同，结果与原实现逐位一致。
- 对 self-attention 的 q、k，写出之前顺便做 RoPE，省掉单独的 RoPE kernel。
- k 的结果直接写到 KV cache 中它最终所在的位置（见 2.4 节），不需要再拷贝。

这样，self-attention 的 q、k 两次 RMSNorm 加 RoPE 从每层 3.08 ms 降到 0.34 ms；cross-attention q 的 RMSNorm 从 1.39 ms 降到 0.14 ms。

### 2.3 残差、LayerNorm 和 modulate 融合

每层有 3 个 LayerNorm，前后都跟着一些单独执行的逐元素运算：

| 位置 | 原始实现中依次执行的运算 |
| --- | --- |
| self-attention 之前 | 上一层 FFN 输出乘以 gate → 加回残差 → LayerNorm → modulate |
| cross-attention 之前 | self-attention 输出乘以 gate → 加回残差 → LayerNorm |
| FFN 之前 | cross-attention 输出加回残差 → LayerNorm → modulate |

其中每一步都要把 18 MB 的张量完整地读写一遍。

现在用一个融合 kernel 一次完成"加回残差 → LayerNorm → modulate"：

- 加完残差的 hidden state 后面还要用，所以在 kernel 里顺便把它写回；乘以 gate 和 modulate 所需的参数，按 2.1 节的段索引直接读取。
- 同样为了保证逐位一致，均值和方差的计算方式（每个线程的累加顺序、线程间的合并顺序）与 PyTorch 原实现的 LayerNorm 完全相同。
- 三个位置共用这一个 kernel，只是开启的功能不同（是否乘 gate、是否 modulate）。

此外，每层末尾原本还有一步"FFN 输出乘以 gate 再加回残差"。现在不单独执行这一步，而是把它交给下一层开头的融合 kernel 一起做，这样又省掉一次完整的读写。只有最后一层的这一步需要单独执行。

最终，self-attention 之前的这一段（含上一层末尾的残差）从每层 0.85 ms 降到 0.32 ms，FFN 之前的一段从 0.54 ms 降到 0.33 ms，cross-attention 之前的一段从 0.53 ms 降到 0.50 ms，收益较小。

### 2.4 零拷贝 KV cache

DreamZero 逐 chunk 生成视频，每个 chunk 的 self-attention 不仅要看当前 chunk 的 token，还要看之前若干 chunk 的 K/V。这些历史 K/V 保存在 KV cache 中。

原始实现中，KV cache 和送进 attention 的 K/V 是两块不同的内存。每次前向，每一层都要先把"历史 K/V、当前视频 token 的 K/V、当前 action/state token 的 K/V"三部分拷贝到一块临时内存中拼起来，再交给 attention。每层要拷贝 6 次（K、V 各 3 段），约 0.73 ms。

现在调整了 KV cache 的内存布局，让 attention 直接读取 cache 本身：

- 每层的 cache 多预留一次前向所需的空间。前面存放历史 K/V，紧接着就是当前前向的 K/V。
- 当前前向算出的 K/V 直接写到这个位置：V 由 QKV 投影的 GEMM 直接写入，K 由 2.2 节的 RMSNorm + RoPE kernel 直接写入。
- 这样历史 K/V 和当前 K/V 在内存中本来就是连续的，attention 直接读取这一段即可，不需要任何拷贝。
- 需要把当前 chunk 加入 cache 时，因为数据已经在正确的位置，只需更新记录的长度。只有当 cache 超出滑动窗口大小时，才需要把保留的部分整体前移一次。action/state 的 K/V 不会保存进 cache，下次前向直接覆盖即可。

这样每层 0.73 ms 的拷贝就完全消除了。

### 2.5 效果

下表的数据与第一节的测量方式相同：

- 测量对象是第 6 个 chunk 中的第 2、3、4 个 denoise step。
- 一个 denoise step 中，40 层 transformer block 在 cond、uncond 两个 CFG 分支上各执行一次，共 80 次 block 执行。表中的数字是"一次 block 执行"的平均耗时：先把某项运算在一个 step 内所有 80 次 block 执行中的 GPU 时间加起来，对 3 个 step 取平均，再除以 80。


| 优化 | 涉及的运算 | 优化前 | 优化后 |
| --- | --- | ---: | ---: |
| AdaLN 只保存不同的行（2.1） | 生成调制参数 | 1427.4 | 3.0 |
| RMSNorm + RoPE 融合（2.2） | self-attention q、k 的 RMSNorm 和 RoPE | 3081.3 | 335.1 |
| RMSNorm 融合（2.2） | cross-attention q 的 RMSNorm | 1392.7 | 139.2 |
| 零拷贝 KV cache（2.4） | 拼接 K/V 的拷贝 | 729.0 | 0 |
| 融合 + 推迟残差（2.3） | self-attention 之前（含上一层末尾的残差） | 853.8 | 320.7 |
| 融合（2.3） | cross-attention 之前 | 525.3 | 495.5 |
| 融合（2.3） | FFN 之前 | 538.7 | 325.0 |
| — | 两路 cross-attention 结果相加（未改动） | 168.1 | 167.4 |
| — | GELU（并入 FFN 第一个 GEMM，属于 GEMM 部分的改动） | 388.3 | — |
| | **合计** | **9104.6** | **1785.9** |

- 每个 block 中访存型的 kernel 和拷贝从 47 个减少到 8 个，耗时下降 5.1 倍。
- 一个 denoise step 中，block 内访存型算子的总耗时从 728 ms 降到 143 ms，节省约 585 ms。

## 三、优化 GEMM

> **背景：Thor 上的 BF16 GEMM 同时受功耗和内存带宽限制**：Thor 的 GPU 有 20 个 SM 和 32 MB L2 cache，按 1.575 GHz 计算，BF16 稠密算力峰值约 258 TFLOPS，但实际能达到的远低于此，主要有两个原因：
>
> - **功耗**：GPU 的供电轨有电流和功率上限（约 94 W）：GEMM 连续满负荷运行时会触发过流保护，硬件通过跳过时钟周期降频，实际 SM 频率只剩 0.87 GHz 左右。
> - **内存带宽**：内存带宽只有约 273 GB/s，算力与带宽之比约为 945 FLOP/字节，即每从内存读入 1 字节，需要做约 945 次运算才能让 Tensor Core 不空等。以 1785 × 5120 × 5120 的投影为例，输入、权重、输出合计约 89 MB，即使每个字节只读写一次，每字节也只对应约 1050 次运算，刚刚超过这个平衡点。而 L2 只有 32 MB，放不下全部操作数，部分数据不得不重复从内存读取，GEMM 很容易变成受内存带宽限制。K 更大的 FFN 第二层更是明显受带宽限制（见 3.3 节）。
>
> 因此在真实推理中，当输入需要从内存读取时，BF16 GEMM 的吞吐量约为 115～125 TFLOPS；输入已在 L2 中时可以更高（见 3.1 节）。孤立地反复跑一个 GEMM 得到的数字往往与真实推理差别很大，本节的数据都来自完整推理中的测量。

原始版本中，每个 transformer block 有 8 个 GEMM（每个 CFG 分支 1785 行输入）：

| GEMM | 形状（M × N × K） | 计算量 |
| --- | --- | ---: |
| Q、K、V 投影（3 次） | 1785 × 5120 × 5120 | 各 93.6 GFLOP |
| self-attention 输出投影 | 1785 × 5120 × 5120 | 93.6 GFLOP |
| cross-attention Q 投影 | 1785 × 5120 × 5120 | 93.6 GFLOP |
| cross-attention 输出投影 | 1785 × 5120 × 5120 | 93.6 GFLOP |
| FFN 第一层（up） | 1785 × 13824 × 5120 | 252.7 GFLOP |
| FFN 第二层（down） | 1785 × 5120 × 13824 | 252.7 GFLOP |

它们全部调用 cuBLAS，吞吐量只有 86～99 TFLOPS，一个 denoise step 中 block 内的 GEMM 共耗时约 912 ms。

### 3.1 注意力相关的投影：CUTLASS kernel

Q、K、V、self-attention 输出、cross-attention Q、cross-attention 输出这 6 个投影的形状都是 1785 × 5120 × 5120。它们统一换成了一个用 CUTLASS 生成的 kernel：

- 每个 tile 计算 256 × 256 的输出块，K 方向每次取 64。两个相邻 SM 组成一个 cluster，共同完成一个 tile（Blackwell 的 2-SM MMA），各自只需加载一半的操作数。
- bias 在 epilogue 中直接加上：在 FP32 累加结果上加 bias，再转成 BF16 写出。
- K 方向严格按顺序累加，不做 split-K。
- 输出可以直接写到指定位置。例如 V 投影的结果直接写进 KV cache（见 2.4 节），省掉一次拷贝。

结果：QKV 三个投影从每层 2.99 ms 降到 2.47 ms，self-attention 输出投影从 0.95 ms 降到 0.62 ms，cross-attention 的 Q 投影从 0.97 ms 降到 0.90 ms，输出投影从 0.98 ms 降到 0.59 ms。

可以看到，同一个 kernel 在 block 中不同位置的吞吐量差别很大：紧跟在 attention 之后的两个输出投影约 151～159 TFLOPS，cross-attention Q 投影只有约 104 TFLOPS；连续执行的三个 QKV 投影一个比一个慢（约 119 → 114 → 108 TFLOPS）。之前另外几次完整推理的 trace 中也是同样的规律。原来的 cuBLAS kernel 在各个位置都在 92～99 TFLOPS 之间，没有这种差别。

进一步分析发现，这个差别主要来自 L2 cache：

- **输入是否还在 L2 中**：1785 × 5120 的输入矩阵约 18 MB，能放进 32 MB 的 L2。同一个 5120 × 5120 投影，在满频率下输入已在 L2 中时耗时约 485 µs，需要从内存读取时约 615 µs。用 Nsight Compute 在完整推理中测量也确认，每个 GEMM 的输入只从内存读一次，差别就在于这一次读取能否命中 L2。
- **两个输出投影为什么快**：它们的输入是 attention 的输出，attention 刚刚写完，数据还在 L2 中。
- **QKV 和 cross-attention Q 为什么慢**：它们的输入来自 2.3 节的融合 LayerNorm kernel。这个 kernel 同时写出两个张量：加完残差的 hidden state 和归一化后的结果，合计约 36 MB，超过了 L2 的容量。后写的残差会把归一化结果挤出 L2，紧接着的 GEMM 只能从内存重新读取输入。FFN 第一层的输入同样来自这个 kernel，也受到同样的影响。三个 QKV 投影一个比一个慢，还叠加了功耗的影响：连续的满负荷 GEMM 越往后越容易触发过流降频，实测 K、V 投影运行时的实际 SM 频率只有 0.9～1.1 GHz。

**改进**：融合 LayerNorm kernel 写出残差时加上"优先淘汰"（evict first）的 L2 缓存提示。残差要到下一个融合 kernel 才会再用到，没有必要留在 L2 中；这样归一化结果就能留在 L2 里，供紧随其后的 GEMM 直接读取。这只是一个缓存提示，计算结果逐位不变。完整推理中，Q 投影从约 119 TFLOPS 提高到约 144 TFLOPS，cross-attention Q 投影从约 104 TFLOPS 提高到约 140 TFLOPS，FFN 第一层也快了约 9%；但 K、V 投影仍受功耗限制，提升不大。

### 3.2 FFN 第一层：GEMM 与 GELU 融合

原始实现中，FFN 第一层先用 cuBLAS 算出 1785 × 13824 的 BF16 结果，再由一个单独的 kernel 计算 GELU。这个中间结果约 49 MB，要先写出、再读回，GELU kernel 每层耗时约 0.39 ms。

现在用一个 CUTLASS kernel 完成全部计算：

- 主循环与 3.1 节相同（256 × 256 的 tile，2-SM cluster），流水线固定为 5 级。
- epilogue 在 FP32 累加结果上依次加 bias、计算 tanh 近似的 GELU，最后只做一次 BF16 舍入后写出。中间结果不再经过显存。

结果：GEMM 加 GELU 从每层 2.96 ms（GEMM 2.57 ms + GELU 0.39 ms）降到 2.19 ms。

由于省掉了 GELU 之前的那次 BF16 舍入，计算结果与原实现不再逐位一致，相对误差（RMS）约为 0.17%，在 BF16 的精度范围内。

### 3.3 FFN 第二层：down GEMM

FFN 第二层（down）的形状是 1785 × 5120 × 13824。它的特殊之处在于 K 很大：输入激活约 49 MB，权重约 141 MB，两者都远大于 32 MB 的 L2，而输出只有 18 MB。分析发现，这个 GEMM 的瓶颈是从内存读取操作数，而不是 Tensor Core 的计算：每次调用实际要从内存读入 380～450 MB，Tensor Core 只有约 39% 的时间在工作。用 3.1 节那种 256 × 256 的 CUTLASS kernel 反而更差，重复读取的数据更多。因此这里针对这个形状和 Thor 的内存特性，对 kernel 的结构与各项参数逐一做了细致的调优，目标是提高 L2 命中率和加快内存读取。

**整体结构**
- kernel 是常驻的（persistent）：20 个 SM 两两组成 10 对，整个计算过程中一直运行，每对依次处理分到的输出 tile。
- 每个 tile 的大小是 256 × 512：两次 256 宽的 MMA 共用同一块输入激活，这样读入一次激活能多算一倍的输出。累加结果放在 Tensor Memory 中（Blackwell 新增的、专门存放 MMA 累加结果的片上存储），512 列正好用满。
- 全部输出被切成 70 个 tile（7 行 × 10 列），每对 SM 处理 7 个。tile 的执行顺序和每个 tile 在 K 方向上是正向还是反向累加，都写在一张静态表中。这张表是在真实的"FFN 第一层 → 第二层"连续执行的场景下，通过爬山搜索找到的，并用一个 L2 缓存模拟器验证（模拟器预测的内存读取量与实测相差约 1%）。反向累加的作用是：一个 tile 结束时刚读入的数据，可以被下一个 tile 一开始就用上，此时它们还在 L2 中。

**针对 Thor 内存的优化**
- **每次连续读取 256 字节**：实测发现，Thor 的 LPDDR 内存在每行只读 128 字节时带宽约 145 GB/s，而每行连续读 256 字节时可达约 200 GB/s。内存频率升到最高后两者都约 224 GB/s，但内存频率由系统动态调节，推理中经常达不到最高。常规写法中，数据搬运单元（TMA）每次只取一行中的 128 字节；这里调整了共享内存中的数据排布，让每次搬运一次取到同一行的连续 256 字节，同时仍满足 Tensor Core 对数据排布的要求。为此，K 方向每次取 128，流水线为 2 级。
- **尽早释放累加器**：原来 epilogue 要等所有结果都写回显存，才释放 Tensor Memory，下一个 tile 的 MMA 在 tile 交界处要空等约 22 µs。现在 epilogue 先把累加结果加上 bias、转成 BF16 放进寄存器，立即释放 Tensor Memory，再慢慢写回显存，写回与下一个 tile 的计算重叠，空等降到约 5 µs。

**没有效果的尝试**

- **把激活留在 L2**：曾尝试把激活标记为优先保留在 L2 中，但一组激活约 35 MB，超过了 L2 容量，反而把其他数据挤了出去；改回普通优先级后更快。
- **L2 预取**：各种形式的 L2 预取（让 TMA 或单独的 warp 提前把数据读进 L2）都使 kernel 变慢。
- **split-K**：让一半激活常驻 L2 的 split-K 方案虽然减少了权重的读取，但部分和的写出会把常驻的数据挤出 L2，整体更慢。

结果：down GEMM 从每层 2.94 ms 降到 2.07 ms，吞吐量从 86 TFLOPS 提高到 122 TFLOPS。

反向累加改变了部分 tile 中浮点加法的顺序，所以结果与 cuBLAS 不是逐位一致，约 0.01% 的输出元素相差 1 个 BF16 最低有效位。

### 3.4 效果

一个 block 跑一个 CFG 分支时，各 GEMM 的耗时（统计方式与 2.5 节相同：第 6 个 chunk 中第 2～4 个 denoise step 的平均，再除以 80 次 block 执行）：

| GEMM | 优化前 (µs) | 吞吐量 (TFLOPS) | 优化后 (µs) | 吞吐量 (TFLOPS) |
| --- | ---: | ---: | ---: | ---: |
| Q、K、V 投影（3 次合计） | 2985.5 | 94.0 | 2470.3 | 113.7 |
| self-attention 输出投影 | 950.8 | 98.4 | 619.4 | 151.1 |
| cross-attention Q 投影 | 974.5 | 96.0 | 900.3 | 103.9 |
| cross-attention 输出投影 | 979.0 | 95.6 | 589.5 | 158.8 |
| FFN 第一层 | 2568.7 | 98.4 | 2191.9（含 GELU） | 115.3 |
| FFN 第一层之后的 GELU | 388.3 | — | 0（已融合） | — |
| FFN 第二层 | 2936.5 | 86.0 | 2066.7 | 122.3 |
| **合计** | **11783.3** | | **8838.1** | |

- 每个 block 中 GEMM（含原来单独的 GELU）的耗时下降约 25%，从 11.8 ms 降到 8.8 ms。
- 一个 denoise step 中，block 内 GEMM 的总耗时从 912 ms 降到 707 ms，再加上省掉的 GELU（约 31 ms），共节省约 236 ms。

## 四、优化 Attention

第一节的剖析显示，attention 约占一个 denoise step 的 16%：每层一次 self-attention（使用 FA2，约 3.2 ms）和两次 cross-attention（文本、图像各一次，合计约 0.69 ms）。

| | query 长度 | key/value 长度 | 说明 |
| --- | ---: | ---: | --- |
| self-attention | 1785 | 历史帧 + 当前 chunk，最多约 7900 | 随 KV cache 增长，越往后越长 |
| cross-attention（文本） | 1785 | 512 | 文本 token |
| cross-attention（图像） | 1785 | 257 | CLIP 图像 token |

### 4.1 self-attention：FlashAttention 2 换成 FlashAttention 4

DreamZero 原本的 self-attention 使用 FlashAttention 2（FA2）。FA2 是为 Ampere 架构设计的，在 Thor 上只能使用老一代的 Tensor Core 指令，用不到 Blackwell 新增的硬件特性。FlashAttention 4（FA4）是专门为 Blackwell 编写的版本：它使用 Blackwell 的新一代 Tensor Core 指令（tcgen05 MMA），把中间结果放在 Tensor Memory 中，用 TMA 搬运数据，并让矩阵乘法、softmax 和数据搬运在不同的 warp 上流水线式地重叠执行。

DreamZero 的 self-attention 每次只有一个序列（batch size 为 1），可以直接调用 FA4。因此 self-attention 改为调用 FA4，只需要在配置中切换 attention 的实现。

以下测量在优化后的版本上进行，只切换 self-attention 的实现，因此 FA2 一列的耗时与第一节的原始版本不同。GPU 频率上限为默认的 1575 MHz。每种配置完整推理 3 次，不同配置交错运行，以免温度和功耗状态的变化偏向某一种配置；kernel 耗时来自 chunk 6 第 2～4 个 DiT step 的 GPU trace（方法同第一节）。

| | FA2 | FA4 | 变化 |
| --- | ---: | ---: | ---: |
| 单次 self-attention kernel | 3220 µs | 1393 µs | 快 2.3 倍 |
| 每个 denoise step 中 self-attention 合计 | 257.6 ms | 111.4 ms | −146 ms |
| denoise step 耗时（chunk 5～10 稳态均值） | 1105 ms | 966 ms | −12.6% |
| Run DiT | 123.5 s | 116.8 s | −6.7 s |

（表中 1785 个 query 对 7920 个 key/value；每个 step 有 40 层 × 2 个 CFG 分支 = 80 次 self-attention。）

kernel 上省下的 146 ms 与 denoise step 实际缩短的约 140 ms 基本一致。

换成 FA4 后输出不再与 FA2 版本逐位相同：action 最大绝对误差 0.160、平均绝对误差 0.006，视频整体 PSNR 27.1 dB。原因与 4.2 中分析的相同，是 BF16 计算顺序的差异经自回归累积放大的结果。相对于 12.6% 的加速，这个代价是可以接受的。

### 4.2 cross-attention：尝试 FA4，未采用

cross-attention 也尝试换成了 FA4（文本、图像两次 cross-attention 都换），self-attention 保持 FA4，其余配置完全相同。测量方式与 4.1 相同。

**速度**：单看 kernel，FA4 同样更快：

| | FA2 | FA4 |
| --- | ---: | ---: |
| 单次 cross-attention（文本，512 个 key） | 462 µs | 206 µs |
| 单次 cross-attention（图像，257 个 key） | 285 µs | 195 µs |
| 每个 denoise step 中 cross-attention 合计 | 59.7 ms | 32.0 ms |

每个 denoise step 节省约 28 ms，约占 step 耗时的 3%，但是由于其他 kernel 的耗时波动，这一部分的耗时情况没有体现到端到端耗时上。

**精度**：输出与 FA2 版本相比的偏差：

| 指标 | 结果 |
| --- | --- |
| action 最大绝对误差 | 0.237（action 取值范围约 −2.8～2.0） |
| action 平均绝对误差 | 0.010 |
| 视频整体 PSNR | 25.6 dB |
| 逐帧 PSNR | 第 1 帧 43.5 dB，第 3 帧 36.2 dB，第 67 帧降到 16.2 dB |

FA4 与 FA2 内部的分块方式和计算顺序不同，单次 attention 的结果只在 BF16 精度范围内有微小差异。但 DreamZero 逐 chunk 自回归地生成：每个 chunk 的输出会进入 KV cache，影响后面所有 chunk。这些微小差异因此被一路累积放大：前几个 chunk 的 action 最大误差只有 0.01 左右，到第 8 个 chunk 已经增大到 0.24，视频后段部分帧的 PSNR 也降到 20 dB 以下。

**Trade-off：** cross-attention 的 key/value 只有几百个 token，计算量很小。换成 FA4 后每个 step 只能节省约 28 ms（约 3%），输出偏差却比只换 self-attention 时更大（action 最大误差 0.237 对 0.160，PSNR 25.6 dB 对 27.1 dB）。最终没有采用，cross-attention 仍使用 FA2。

## 五、注意事项

### 5.1 测性能前先锁定频率

Thor 的 GPU 频率和内存（EMC）频率默认都由系统动态调节（DVFS）：GPU 空闲或负载较轻时会降频，内存频率也会在约 665 MHz 到 4266 MHz 之间变化。在这种状态下测得的耗时，很大程度上取决于测量时频率恰好处在哪一档，而不是代码本身的快慢。例如，内存频率有时会在整个进程运行期间停在 3200 MHz，而不是最高的 4266 MHz，此时受内存带宽限制的 kernel 会慢约 30%。

因此，对比优化前后的性能时，一定要先把频率固定下来，并保证对比的两个版本运行在完全相同的频率设置下。在 Thor 上可以这样做：

```bash
sudo nvpmodel -m 0          # 切换到 MAXN 功耗模式
sudo jetson_clocks --store  # 先保存当前的频率设置，便于测完恢复
sudo jetson_clocks          # 把 CPU、GPU 和内存频率固定在当前模式的最高值
sudo jetson_clocks --show   # 确认当前频率
# 测试结束后恢复
sudo jetson_clocks --restore
```

也可以直接查看 GPU 和内存的频率：

```bash
cat /sys/class/devfreq/gpu-gpc-0/cur_freq   # GPU 频率
cat /sys/class/devfreq/bwmgr/cur_freq       # 内存（EMC）频率
```

需要注意的是，锁定频率并不能阻止第三节提到的过流降频：GEMM 连续满负荷运行时，硬件仍会通过跳过时钟周期来降低实际频率，而上面读到的频率值看不出这一点，需要查看实际频率。

过流降频还带来一个现象：GPU 空闲一段时间（例如两次测量之间有停顿）之后，紧接着运行的 GEMM 会明显更快。这是因为空闲期间过流保护得以恢复，之后的约 150 ms 内 GEMM 能以接近满频率运行，随后又回到降频状态。这个现象同样不会因为锁定频率而消失，所以测量时应让被测程序连续运行，避免在测量过程中插入停顿（见 5.2 节）。

### 5.2 profiler 本身会改变测量结果

在 Thor 上，用 PyTorch profiler 抓 trace 会明显影响被测程序的速度，而且影响的方向取决于抓取方式：

- **同时记录 CPU 端事件会让推理变慢**：记录 CPU 端的函数调用时，同一个 denoise step 比不抓 trace 时慢了约 30%。
- **在两个 step 之间保存 trace 会让下一个 step 变快**：保存 trace 需要 1～2 秒，这段时间 GPU 处于空闲状态。紧接着的下一个 step 中，GEMM 明显变快，最多快了约 45%，测得的吞吐量远高于正常推理时的水平。

这两种情况得到的分项耗时都不能代表真实推理。本文最终采用的做法是：

- 只记录 GPU kernel，不记录 CPU 端事件；
- 一次连续抓取多个相邻的 step，全部结束后再保存 trace，中间不让 GPU 空闲；
- 每个 kernel 属于哪一部分，通过与一次带标签的参考运行对照 kernel 序列来确定。

这样抓 trace 时 step 只比不抓时慢约 3%。

## 六、结果

本节对比优化前的原始版本与当前版本。原始版本的 self-attention 使用 FA2，当前版本使用 FA4（4.1 节）；两者的 cross-attention 都使用 FA2。GPU 频率上限均为默认的 1575 MHz，分项耗时的测量方法与第一节相同。原始版本的数据来自 2 次运行（端到端、分项耗时各 1 次），当前版本的端到端数据为 3 次运行的平均，6.2、6.3 中当前版本的分项耗时来自 1 次运行。

### 6.1 端到端耗时

| 阶段 | 原始版本 | 当前版本 | 加速 |
| --- | ---: | ---: | ---: |
| 模型加载 | 63.4 s | 63.3 s | — |
| Run DiT（10 个 chunk） | 212.9 s | 116.8 s | 1.82× |
| 总耗时 | 277.6 s | 181.5 s | 1.53× |
| 单个 denoise step（chunk 5～10 稳态均值） | 1926 ms | 966 ms | 1.99× |
| 单个 denoise step（全部 80 步均值） | 2113 ms | 969 ms | 2.18× |

全部 80 步均值的加速比稳态更大，是因为原始版本在第一个 segment（chunk 1～4）处于预热阶段，每个 step 比稳态慢约 0.4～0.5 s（见 1.3 节）；当前版本没有明显的预热阶段，从第一个 chunk 起就基本稳定。

### 6.2 一个 denoise step 的耗时

第 6 个 chunk，第 2～4 个 DiT step 的平均，单位 ms：

| 部分 | 原始版本 | 当前版本 |
| --- | ---: | ---: |
| DiT blocks（40 层 × 2 个 CFG 分支） | 1951.4 | 979.1 |
| 其余部分（pre-infer、video head、action decoder、CFG 合成、scheduler 更新） | 4.7 | 3.8 |
| **合计（GPU kernel 时间）** | **1956.1** | **983.0** |
| 实测 step 墙钟时间（含 profiler 开销） | 1967 | 989 |

### 6.3 一个 transformer block 的耗时

一个 block 跑一个 CFG 分支的耗时，单位 µs，统计方式与 1.5 节相同，按原始版本耗时从高到低排列：

| 部分 | 原始版本 | 当前版本 | 加速 | 对应的优化 |
| --- | ---: | ---: | ---: | --- |
| self-attention | 3203.7 | 1392.9 | 2.30× | FA2 换成 FA4（4.1） |
| q/k RMSNorm + RoPE | 3081.3 | 359.3 | 8.58× | RMSNorm + RoPE 融合（2.2） |
| QKV 投影 | 2985.5 | 2309.4 | 1.29× | CUTLASS 2-SM GEMM（3.1） |
| FFN up + GELU | 2957.0 | 2003.8 | 1.48× | GEMM 与 GELU 融合（3.2） |
| FFN down | 2936.5 | 2123.2 | 1.38× | 调优的 down GEMM（3.3） |
| cross-attn Q 投影 + RMSNorm | 2367.2 | 814.4 | 2.91× | CUTLASS GEMM（3.1）+ RMSNorm 融合（2.2） |
| AdaLN 调制参数 | 1427.4 | 3.0 | 476× | 只保存不同的行（2.1） |
| cross-attn O 投影 | 979.0 | 566.9 | 1.73× | CUTLASS 2-SM GEMM（3.1） |
| self-attn O 投影 | 950.8 | 637.5 | 1.49× | CUTLASS 2-SM GEMM（3.1） |
| cross-attention（text + image） | 857.1 | 918.0 | — | 未改动（FA2） |
| norm1 + modulate + 上一层的 FFN gate 残差 | 853.8 | 335.9 | 2.54× | 融合 + 推迟残差（2.3） |
| KV cache 拼接 / 写回 | 729.0 | 0 | — | 零拷贝 KV cache（2.4） |
| 残差 + norm2 + modulate | 538.7 | 302.4 | 1.78× | 融合（2.3） |
| gate 残差 + norm3 | 525.3 | 472.5 | 1.11× | 融合（2.3） |
| **合计** | **24392.5** | **12239.2** | **1.99×** | |

说明：

- 原始版本中「FFN gate 残差」（418.5 µs）和「norm1 + modulate」（435.3 µs）是两项。当前版本把 FFN gate 残差推迟到下一层的 norm1 中一起完成（2.3 节），因此这里合并为一行。