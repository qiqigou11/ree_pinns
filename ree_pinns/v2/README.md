# 风化壳型稀土矿床 PINN 模型 v2

## 概述

基于 Physics-Informed Neural Networks (PINNs) 的风化壳型稀土矿床迁移模型，整合了文献实测数据。

## 数据来源

使用 Nagasawa2024 花岗岩风化壳剖面数据：
- 样品数：42
- 深度范围：0 - 7.8 m
- LREE 范围：69.6 - 304.5 ppm
- HREE 范围：17.5 - 149.9 ppm
- pH 范围：4.9 - 6.9

## 物理方程

```
∂C/∂t = D·∂²C/∂z² - [u_inf - u_cap(t)]·∂C/∂z - k_ads(pH)·C + k_weathering·(C_parent - C)
```

### 方程项说明

| 项 | 含义 |
|----|------|
| `D·∂²C/∂z²` | 扩散项 |
| `[u_inf - u_cap(t)]·∂C/∂z` | 净对流项（渗透 - 毛细上升） |
| `k_ads(pH)·C` | pH 依赖吸附项 |
| `k_weathering·(C_parent - C)` | 风化释放项 |

## 模型改进 (v1 → v2)

1. **网络架构**：残差连接 + GELU 激活
2. **物理方程**：pH 依赖吸附项 + 季节性对流调制
3. **训练策略**：自适应权重 + 梯度裁剪
4. **数据集成**：文献实测剖面数据

## 文件说明

| 文件 | 说明 |
|------|------|
| `ree_pinn_v2.py` | 主程序 |
| `results_v2.png` | 训练结果可视化 |
| `model_v2.pth` | 训练后的模型权重 |

## 使用方法

```bash
cd /Users/suheng/Desktop/claudecode/pinns/ree_pinns/v2
python ree_pinn_v2.py
```

## 模型参数

```
D = 0.0100 m²/year          # 扩散系数
u_inf = 5.00e-05 m/year     # 渗透速度
u_cap_max = 3.00e-05 m/year # 毛细上升速度
k_ads_base_LREE = 0.10 1/year
k_ads_base_HREE = 0.30 1/year
k_weathering = 0.020 1/year
frac_HREE = 1.50            # HREE 分馏系数
```

## 训练结果

- 迭代次数：3000
- 物理损失：~3.66
- 实测数据损失：~0.014
- 边界损失：~0.17

## 下一步改进方向

1. 集成更多剖面数据（Li2019, Fu2019 等）
2. 增加气候变量时空调制
3. 二维地形效应建模
4. 不确定性量化