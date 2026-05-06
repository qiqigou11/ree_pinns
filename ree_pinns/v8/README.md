# REE PINN v8 - Per-Profile Modeling

## 核心思路
按文献分组建模，每个剖面独立训练。

## 根因分析（v2~v7 失败原因）
| 版本 | 问题 | 根因 |
|------|------|------|
| v2 | 浓度偏小100倍 | 浓度归一化与边界条件C_atm不一致 |
| v3-v4 | MPS梯度断裂 | requires_grad在/操作后丢失 |
| v5 | log(C)崩溃 | log变换过激进，网络找不到正确量级 |
| v6 | IC无意义 | IC loss(C=0)与稳态物理矛盾 |
| v7 | 剖面量级混乱 | 全局C_scale对不同文献失效 |

## v8方案
1. 按文献分组建模，每篇独立C_scale归一化
2. 删除IC loss，只保留有物理意义的损失项
3. 1D SIREN网络，输入z'=z/z_max，输出ppm

## 网络架构
z' -> [Sine(Linear)]x4 -> [Linear+Sigmoid] -> C_scale x ppm

## 损失函数
L_total = 100 x L_data + 10 x L_bc

## 结果
| 文献 | R2_L | R2_H | 预测范围 | 状态 |
|------|------|------|---------|------|
| 1 Nagasawa | 0.997 | 0.999 | 71~303ppm | OK |
| 2 Li2019 | 0.997 | 0.939 | 68~192ppm | OK |
| 3 LiZhou | 0.969 | 0.952 | 114~830ppm | OK |
| 4 Fu2019 | 0.994 | 0.987 | 352~1353ppm | OK |
| 5 Yaraghi | 0.994 | 0.969 | 240~758ppm | OK |
| 6 Luo | 0.406 | 0.390 | 99~679ppm | FAIL |

平均R2_L: 0.893

## 文件
- ree_pinn_v8.py - 主程序
- model_v8.pt - 模型权重
- results_v8.png - 结果图

## 运行
cd ~/Desktop/claudecode/pinns/ree_pinns/v8
~/anaconda3/bin/python ree_pinn_v8.py
