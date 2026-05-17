# REE PINN 项目演进与代码逻辑梳理

## 1. 项目概述

**目标**：构建物理信息神经网络（PINN）模拟风化壳剖面中稀土元素（REE）的分馏过程。

**核心问题**：风化壳中轻稀土（LREE）和重稀土（HREE）因对粘土矿物吸附亲和力不同而产生分馏，需建立物理模型描述这一现象。

---

## 2. 版本演进

### 2.1 版本对比总表

| 版本 | 方法 | 输入 | 输出 | 物理约束 | 关键创新 |
|------|------|------|------|----------|----------|
| **v2** | SIREN + 残差连接 | [z, T, P, pH] | TotalREE | 无 | pH依赖吸附、季节性对流调制 |
| **v8** | SIREN（每剖面独立） | [z_norm] | TotalREE | 无 | 每剖面独立训练、SIREN激活 |
| **v9** | SIREN（多剖面集成） | [z, T, P, R] 4D | TotalREE | 无 | 气候数据集成、4维输入 |
| **v10** | SIREN（等权重） | [z, T, P, R] 4D | TotalREE | 无 | **剖面级损失平均**（解决Nagasawa主导问题） |
| **v14** | scipy曲线拟合 | - | λ_L, λ_H | 解析解 | **首次分离L/H**、解析PDE解 |
| **v15** | 混合NN+解析 | [z, T, P, R, clay] 5D | λ_L, λ_H | λ_H > λ_L | **NN预测λ**、粘土矿物映射、LOO-CV |

### 2.2 关键版本详细说明

#### v10 — 数据驱动基准模型
```
输入: [z_norm, T_norm, P_norm, R_norm] (4D)
输出: TotalREE (ppm)
损失: Per-profile MSE平均
网络: 5层SineLayer, 64神经元
```

**问题**：
- 纯数据驱动，缺乏物理意义
- 无法解释L/H分馏机制
- Nagasawa有42条样品，其他合计58条，样品级损失会让Nagasawa主导

**解决v10**：
- 损失按剖面计算再平均，每个剖面等权重

---

#### v14 — 物理框架建立（传统拟合）
```python
# 稳态对流-弥散-吸附方程
D·C''(z) - v·C'(z) - k·(C - C_parent) = 0

# 解析解
C(z) = C_parent + (C_atm - C_parent) · exp(-λ·z)

# 特征衰减长度
λ = (v/2D) · [1 + sqrt(1 + 4kD/v²)]
```

**关键创新**：
1. 解析解自动满足边界条件（z=0时C=C_atm，z→∞时C=C_parent）
2. 用scipy.optimize曲线拟合求λ_L和λ_H
3. LREE和HREE有独立的λ值（代表不同的吸附系数k）

**物理机制**：
- λ越大 → 吸附系数k越大 → 更快趋于母岩浓度
- HREE吸附更强 → λ_H > λ_L
- 浅层L/H比值最大，随深度减小

---

#### v15 — 混合预测模型（当前最佳）
```
输入: [z_norm, T_norm, P_norm, R_norm, clay_norm] (5D)
输出: [λ_L, λ_H]
网络: 3层MLP, 32神经元, Softplus输出
```

**关键创新**：
1. **NN预测λ参数**（而非直接预测浓度）
2. **粘土矿物映射**：Profile 3比例模板映射到所有剖面
3. **物理约束损失**：λ_H > λ_L 软惩罚
4. **LOO-CV验证**：真实泛化能力评估

---

## 3. 物理化学框架

### 3.1 控制方程
```
D·C''(z) - v·C'(z) - k·(C - C_parent) = 0
```
- D: 弥散系数 (m²/yr)
- v: 对流速度 (m/yr) — 水向下渗透
- k: 吸附速率常数 (1/yr) — 粘土矿物亲和力
- C_parent: 母岩浓度 (z→∞边界)
- C_atm: 大气输入浓度 (z=0边界)

### 3.2 解析解
```
C(z) = C_parent + (C_atm - C_parent) · exp(-λ·z)

λ = (v/2D) · [1 + sqrt(1 + 4kD/v²)]
```

### 3.3 L/H分馏机制
```
λ_H > λ_L  (HREE离子半径小，更容易被粘土吸附)
→ HREE更快趋于母岩浓度
→ L/H比值在浅层最大，随深度减小
```

### 3.4 边界条件
| 边界 | 条件 | 验证 |
|------|------|------|
| z → 0 | C(0) = C_atm | C(0) = C_parent + (C_atm-C_parent)·1 = C_atm ✓ |
| z → ∞ | C(∞) = C_parent | C(∞) = C_parent + (C_atm-C_parent)·0 = C_parent ✓ |

---

## 4. 数据处理

### 4.1 数据源
- **主数据**: `REE_samples_20260424_with_climate.csv` (109样品)
- **元数据**: `REE_literature_20260424.csv` (剖面信息)

### 4.2 剖面信息
| Profile ID | 来源 | 样品数 | T(°C) | P(mm/yr) | 粘土数据 |
|------------|------|--------|--------|-----------|----------|
| 1 | Nagasawa | 42 | 13 | ~1500 | 无 |
| 2 | Li2019 | 7 | 18 | ~1800 | 无 |
| 3 | Li&Zhou | 12 | 18 | ~1800 | **有**(85%) |
| 4 | Fu2019 | 12 | 20 | ~2000 | 无 |
| 5 | Yaraghi | 9 | 24 | ~2500 | 无 |
| 6 | Luo | 20 | 22 | ~1800 | 部分(9%) |
| 7 | Wang | 6 | ~18 | ~1500 | 无 |

### 4.3 归一化参数
```python
Z_MAX  = 33.0       # 最大深度
T_MEAN = 290.62     # K
T_STD  = 3.78
P_MEAN = 1604.84    # mm/yr
P_STD  = 710.27
R_MEAN = 0.521      # m/yr
R_STD  = 0.231
CLAY_MAX = 1.0      # 归一化粘土总量
```

---

## 5. 网络架构

### 5.1 v10 网络结构（SIREN）
```python
class REE_PINN(nn.Module):
    def __init__(self, h=64, n=5):
        self.net = nn.Sequential(
            SineLayer(4, h, True),      # 4D输入
            *[SineLayer(h, h) for _ in range(n-1)],
            nn.Linear(h, 1),
            nn.ReLU6()
        )
        self.head = nn.Sequential(nn.Linear(h, 1), nn.ReLU6())

    def forward(self, x):
        return self.head(self.net(x)) * C_SCALE  # 4000ppm缩放
```

### 5.2 v15 网络结构（LambdaPINN）
```python
class LambdaPINN(nn.Module):
    """
    预测λ_L和λ_H的神经网络

    输入: [z_norm, T_norm, P_norm, R_norm, clay_norm] (5个特征)
    输出: [λ_L, λ_H] (2个正参数)
    """
    def __init__(self, hidden=32, n_layers=3):
        layers = []
        layers.append(nn.Linear(5, hidden))
        layers.append(nn.ReLU())

        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden, hidden))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden, 2))
        layers.append(nn.Softplus())  # 确保λ > 0

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)  # [λ_L, λ_H]
```

### 5.3 架构对比
| 方面 | v10 | v15 |
|------|-----|-----|
| 输入维度 | 4 | 5 (+clay) |
| 隐藏层 | 5 | 3 |
| 每层神经元 | 64 | 32 |
| 激活函数 | Sine | ReLU + Softplus |
| 输出 | TotalREE (ppm) | [λ_L, λ_H] |
| 参数量 | ~25K | ~5K |

**设计理由**：v15数据更少（109样品），需要更小的网络避免过拟合

---

## 6. 损失函数

### 6.1 v10 损失函数
```python
def step(self, N_bc=200):
    # (1) 边界条件损失
    for pid, d in self.profile_data.items():
        C_bc = self.pinn(x_bc)  # z=0处
        lbc_list.append(torch.mean((C_bc - d['atm']) ** 2))
    loss_bc = torch.stack(lbc_list).mean()

    # (2) 数据损失（剖面平均）
    for pid, d in self.profile_data.items():
        loss_data_list.append(torch.mean((C_pred - d['C']) ** 2))
    loss_data = torch.stack(loss_data_list).mean()

    # 总损失
    loss = 100.0 * loss_data + 10.0 * loss_bc
```

### 6.2 v15 损失函数
```python
def compute_loss(lambda_pred, profile_data):
    """
    L_total = 1.0·L_data + 1.0·L_phys + 0.01·L_reg
    """
    # (1) 数据损失（Profile级别MSE）
    for pid, d in profile_data.items():
        C_pred = analytical_C(z, C_atm, C_parent, lam)
        L_data += torch.mean((C_pred - C_obs) ** 2)
    L_data /= n_profiles

    # (2) 物理约束损失
    # 强制 λ_H > λ_L（HREE吸附更强）
    phys_violation = torch.clamp(lam_L - lam_H, min=0) ** 2
    L_phys = phys_violation

    # (3) 正则化损失
    # 防止λ过大或过小
    L_reg = 0
    L_reg += torch.clamp(lam_L - 10.0, min=0) ** 2
    L_reg += torch.clamp(0.01 - lam_L, min=0) ** 2
    L_reg += torch.clamp(lam_H - 10.0, min=0) ** 2
    L_reg += torch.clamp(0.01 - lam_H, min=0) ** 2

    return 1.0 * L_data + 1.0 * L_phys + 0.01 * L_reg
```

### 6.3 损失权重设计
| 版本 | L_data | L_phys | L_reg | L_bc |
|------|--------|--------|-------|------|
| v10 | 100.0 | 0 | 0 | 10.0 |
| v15 | 1.0 | 1.0 | 0.01 | 0 |

**v15设计理由**：
- 数据约束和物理约束同等权重（1:1）
- 正则化权重很小（0.01）防止过拟合
- 无边界损失因为解析解精确满足边界条件

---

## 7. 粘土矿物处理

### 7.1 Profile 3粘土比例（模板）
```python
P3_CLAY_RATIOS = {
    'Kaolinite': 0.68 / 0.854,    # ~0.80
    'Vermiculite': 0.11 / 0.854,  # ~0.13
    'Illite': 0.06 / 0.854,       # ~0.07
}
```

### 7.2 粘土映射算法
```python
def map_clay_to_profile(profile_z, T_C, P_mm):
    """
    对于每个剖面：
    1. 根据气候估算总粘土量
    2. 按Profile 3的比例分配各粘土矿物
    3. 按深度位置返回粘土含量
    """
    # 气候-粘土关系
    climate_score = (max(0, T_C - 10) / 20) * (max(0, P_mm - 500) / 1500)
    total_clay = min(0.9, 0.3 + 0.6 * climate_score)

    # 分配各粘土矿物
    clay_minerals = {
        'Kaolinite': total_clay * 0.80,
        'Vermiculite': total_clay * 0.13,
        'Illite': total_clay * 0.07,
    }

    return clay_minerals
```

---

## 8. 母岩参数化

### 8.1 C_parent设置优先级
1. **优先**：剖面中有实际母岩样品（Horizon含"Parent"）
2. **备选**：使用最深样品

### 8.2 代码实现
```python
parent_mask = grp_orig['Horizon'].str.contains('Parent', case=False, na=False)
if parent_mask.any():
    parent_grp = grp_orig[parent_mask]
    CL_parent = float(parent_grp['CL'].mean())
    CH_parent = float(parent_grp['CH'].mean())
else:
    CL_parent = float(CL[-1])  # 最深样品
    CH_parent = float(CH[-1])
```

### 8.3 有母岩数据的剖面
- Profile 2 (Li2019): 有母岩样品
- Profile 4 (Fu2019): 有母岩样品
- Profile 5 (Yaraghi): 有母岩样品
- Profile 7 (Wang): 有母岩样品

---

## 9. 训练策略

### 9.1 v15 LOO-CV流程
```python
def leave_one_out_cv(profile_data, model_class, model_params):
    """
    留一交叉验证
    对于每个被剔除的剖面：
    1. 用N-1个剖面训练
    2. 在被剔除的剖面上验证
    """
    cv_results = {}

    for leave_pid in profiles:
        train_data = {p: data[p] for p in profiles if p != leave_pid}
        val_data = {leave_pid: data[leave_pid]}

        # 训练模型
        model = model_class(**model_params)
        train(model, train_data)

        # 验证
        results = validate(model, val_data)
        cv_results[leave_pid] = results

    return cv_results
```

### 9.2 优化器配置
```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    patience=500,
    factor=0.5,
    min_lr=1e-6
)
```

---

## 10. 评估指标

### 10.1 主要指标
| 指标 | 描述 | 目标 |
|------|------|------|
| R²(L/H) | L/H比值的决定系数 | > 0.6 |
| R²(LREE) | LREE浓度的决定系数 | > 0.5 |
| R²(HREE) | HREE浓度的决定系数 | > 0.5 |
| CV R² | 交叉验证R²（真实泛化能力） | > 0.5 |

### 10.2 物理约束验证
- λ_H > λ_L 对所有剖面成立
- λ在合理范围 [0.01, 10.0]

---

## 11. 文件结构

```
ree_pinns/
├── data/
│   ├── REE_samples_20260424_with_climate.csv   # 主数据文件
│   ├── REE_literature_20260424.csv
│   └── modern_runoff_v1_interp.nc
├── v2/                            # pH依赖吸附
├── v8/                            # SIREN每剖面独立
├── v9/                            # 气候集成
├── v10/                           # 剖面级损失平均
├── v11/                           # 开发文件夹
│   ├── ree_pinn_v14.py           # 解析拟合（非NN）
│   ├── ree_pinn_v15.py           # 混合NN+解析
│   └── REE_PINN_v15_Implementation_Plan.md
└── v16/                           # 本文档
```

---

## 12. 待解决问题与未来方向

### 12.1 当前问题
1. **CV R² ~0.376** — 简单指数模型可能对复杂真实剖面拟合不足
2. **物理约束未完全满足** — λ_H > λ_L 在部分剖面不成立
3. **C_parent固定** — 未作为可学习参数

### 12.2 可能的改进方向
1. **让C_parent可学习** — 作为另一个网络输出或全局优化参数
2. **更复杂的物理模型** — 考虑时间依赖、非稳态
3. **多任务学习** — 同时预测λ和C_parent
4. **使用更多气候变量** — 考虑季节性、降水强度等

---

## 13. 核心代码片段

### 13.1 解析解计算
```python
def analytical_C(z, C_atm, C_parent, lam):
    """
    解析解: C(z) = C_parent + (C_atm - C_parent)·exp(-λz)
    """
    return C_parent + (C_atm - C_parent) * torch.exp(-lam * z)
```

### 13.2 完整前向计算
```python
def forward_profile(z, T, P, R, clay, pinn):
    # 归一化
    x = torch.tensor([[
        z / Z_MAX,
        (T - T_MEAN) / T_STD,
        (P - P_MEAN) / P_STD,
        (R - R_MEAN) / R_STD,
        clay / CLAY_MAX
    ]])

    # 预测λ
    lam_L, lam_H = pinn(x).squeeze()

    # 解析解
    C_L = C_parent_L + (C_atm_L - C_parent_L) * exp(-lam_L * z)
    C_H = C_parent_H + (C_atm_H - C_parent_H) * exp(-lam_H * z)

    return C_L, C_H
```

---

## 14. 参考文献

- 解析解来源: 稳态对流-弥散-吸附方程
- 粘土矿物数据: Profile 3 (Li&Zhou剖面)
- 气候数据: PI_climate_data.nc, modern_runoff_v1_interp.nc
