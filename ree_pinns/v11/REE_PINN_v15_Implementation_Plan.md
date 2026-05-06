# REE PINN v15 实现计划

## 1. 项目概述

### 目标
构建一个物理信息神经网络（PINN），用于模拟风化壳剖面中稀土元素（REE）的分馏过程，特别是轻稀土（LREE）和重稀土（HREE）的分离建模。

### 核心创新
1. **预测λ参数而非直接预测浓度** - 小数据集更稳定，避免过拟合
2. **LREE/HREE分离建模** - 物理意义的分馏参数
3. **粘土矿物映射** - 基于Profile 3实测数据的比例模板
4. **解析PDE物理约束** - 精确满足边界条件

---

## 2. 物理化学框架

### 2.1 控制方程
稳态对流-弥散-吸附方程：

```
D·C''(z) - v·C'(z) - k·(C - C_parent) = 0
```

其中：
- D = 弥散系数 (m²/yr)
- v = 对流速度 (m/yr) - 水向下渗透
- k = 一阶吸附速率常数 (1/yr) - 粘土矿物亲和力
- C_parent = 母岩浓度 (z→∞边界)
- C_atm = 大气输入浓度 (z=0边界)

### 2.2 解析解
```
C(z) = C_parent + (C_atm - C_parent) · exp(-λ·z)

λ = (v/2D) · [1 + sqrt(1 + 4kD/v²)]
```

### 2.3 L/H分馏机制
```
λ_H > λ_L  (HREE对粘土的吸附更强)
→ HREE更快趋于母岩浓度
→ L/H比值在浅层最大，随深度减小
```

**物理背景**：HREE（重稀土）离子半径较小，更容易被粘土矿物吸附，因此吸附系数k_H > k_L。

### 2.4 边界条件
解析解**自动精确满足**边界条件：

| 边界 | 条件 | 验证 |
|------|------|------|
| z → 0 | C(0) = C_atm | C(0) = C_parent + (C_atm-C_parent)·1 = C_atm ✓ |
| z → ∞ | C(∞) = C_parent | C(∞) = C_parent + (C_atm-C_parent)·0 = C_parent ✓ |

---

## 3. 数据处理

### 3.1 数据源
- **主数据**: `REE_samples_20260424_with_climate.csv` (109样品)
- **元数据**: `REE_literature_20260424.csv` (剖面信息)

### 3.2 样品信息
| Profile ID | 来源 | 样品数 | T(°C) | P(mm/yr) | 粘土数据 |
|------------|------|--------|--------|-----------|----------|
| 1 | Nagasawa | 42 | 13 | ~1500 | 无 |
| 2 | Li2019 | 7 | 18 | ~1800 | 无 |
| 3 | Li&Zhou | 12 | 18 | ~1800 | **有**(85%) |
| 4 | Fu2019 | 12 | 20 | ~2000 | 无 |
| 5 | Yaraghi | 9 | 24 | ~2500 | 无 |
| 6 | Luo | 20 | 22 | ~1800 | 部分(9%) |
| 7 | Wang | 6 | ~18 | ~1500 | 无 |

### 3.3 粘土矿物处理策略

#### Profile 3粘土比例（归一化到sum=1）
```python
P3_CLAY_RATIOS = {
    'Kaolinite': 0.68 / 0.854,    # ~0.80
    'Vermiculite': 0.11 / 0.854,  # ~0.13
    'Illite': 0.06 / 0.854,       # ~0.07
}
```

#### 映射算法
```python
def map_clay_to_profile(profile_z, profile_id, T_C, P_mm):
    """
    对于每个剖面：
    1. 根据气候估算总粘土量（或用默认值0.5）
    2. 按Profile 3的比例分配各粘土矿物
    3. 按深度位置返回粘土含量
    """

    # 气候-粘土关系（简化）
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

### 3.4 归一化参数
```python
Z_MAX  = 33.0       # 最大深度
T_MEAN = 290.62      # K
T_STD  = 3.78
P_MEAN = 1604.84     # mm/yr
P_STD  = 710.27
R_MEAN = 0.521       # m/yr
R_STD  = 0.231
CLAY_MAX = 1.0       # 归一化粘土总量
```

---

## 4. 网络架构

### 4.1 LambdaPINN 网络
```python
class LambdaPINN(nn.Module):
    """
    预测λ_L和λ_H的神经网络

    输入: [z_norm, T_norm, P_norm, R_norm, clay_norm] (5个特征)
    输出: [λ_L, λ_H] (2个正参数)
    """

    def __init__(self, hidden=32, n_layers=3):
        super().__init__()

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

### 4.2 网络参数
- **输入维度**: 5 (z_norm, T, P, R, clay)
- **隐藏层**: 3层
- **每层神经元**: 32
- **总参数量**: ~5,000
- **激活函数**: ReLU + Softplus输出

### 4.3 前向计算
```python
def forward_profile(z, T, P, R, clay, pinn):
    """
    预测REE浓度剖面

    1. 归一化输入
    2. 网络预测λ_L, λ_H
    3. 解析解计算C(z)
    """
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

## 5. 损失函数

### 5.1 总损失
```
L_total = 1.0·L_data + 1.0·L_phys + 0.01·L_reg
```

**设计理由**：平衡权重确保物理约束（L_phys）与数据约束（L_data）同等重要，避免物理约束被数据损失压制。

### 5.2 数据损失（Profile级别MSE）
```python
def profilewise_data_loss(C_pred, C_obs):
    """
    Per-profile MSE，然后平均

    避免样品多的剖面主导训练
    """
    loss_list = []
    for pid in profiles:
        rel_error_L = ((C_L_pred[pid] - C_L_obs[pid]) / (C_L_obs[pid] + 1)) ** 2
        rel_error_H = ((C_H_pred[pid] - C_H_obs[pid]) / (C_H_obs[pid] + 1)) ** 2
        profile_loss = mean(rel_error_L) + mean(rel_error_H)
        loss_list.append(profile_loss)

    return mean(loss_list)
```

### 5.3 物理约束损失
```python
def physical_constraint_loss(lam_L, lam_H):
    """
    强制 λ_H > λ_L（HREE吸附更强）

    L_phys = max(0, λ_L - λ_H)²
    当 λ_L > λ_H 时施加惩罚
    """
    violation = clamp(lam_L - lam_H, min=0) ** 2
    return violation
```

### 5.4 正则化损失
```python
def regularization_loss(lam_L, lam_H):
    """
    防止λ过大或过小

    合理范围: [0.01, 10.0] m⁻¹
    """
    loss = 0
    loss += clamp(lam_L - 10.0, min=0) ** 2
    loss += clamp(0.01 - lam_L, min=0) ** 2
    loss += clamp(lam_H - 10.0, min=0) ** 2
    loss += clamp(0.01 - lam_H, min=0) ** 2
    return loss
```

---

## 6. 训练策略

### 6.1 优化器配置
```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4  # L2正则化
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    patience=500,
    factor=0.5,
    min_lr=1e-6
)
```

### 6.2 训练循环
```python
for epoch in range(max_epochs):
    optimizer.zero_grad()

    # 前向传播
    lambda_pred = model(x)  # [λ_L, λ_H]

    # 计算损失
    loss, L_data, L_phys = compute_loss(lambda_pred, profile_data)

    # 反向传播
    loss.backward()
    clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step(L_data)
```

### 6.3 留一交叉验证（LOO-CV）
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

---

## 7. 评估指标

### 7.1 主要指标
| 指标 | 描述 | 目标 |
|------|------|------|
| R²(L/H) | L/H比值的决定系数 | > 0.6 |
| R²(LREE) | LREE浓度的决定系数 | > 0.5 |
| R²(HREE) | HREE浓度的决定系数 | > 0.5 |
| RMSE(L/H) | L/H比值的均方根误差 | < 20% |
| CV R² | 交叉验证R²（真实泛化能力） | > 0.5 |

### 7.2 物理约束验证
- λ_H > λ_L 对所有剖面成立
- λ在合理范围 [0.01, 10.0]

---

## 8. 可视化

### 8.1 必做图表
1. **剖面拟合图**: C_L(z), C_H(z)与观测值对比
2. **L/H比值剖面**: 展示随深度减小趋势
3. **λ_L vs λ_H散点图**: 应在1:1线**以下**（因为λ_H > λ_L）
4. **CV R²柱状图**: 每个剖面的泛化能力
5. **气候vs λ**: T, P, R与λ的相关性

### 8.2 图表布局
```
图1: λ_L vs λ_H (散点图)
图2: 训练曲线
图3: CV R² (柱状图)
图4-6: 剖面1-3拟合
图7: L/H比值剖面
图8: 气候vs λ
图9: 粘土vs λ
```

---

## 9. 实现文件

### 9.1 文件结构
```
v11/
├── ree_pinn_v15.py          # 主程序
├── model_v15.pt             # 保存的模型
└── results_v15.png          # 结果图
```

### 9.2 代码模块
```python
# ===================== 数据处理 =====================
load_and_prepare_data()       # 加载CSV，处理缺失值
map_clay_to_profile()         # 粘土矿物映射

# ===================== 网络 =====================
class LambdaPINN(nn.Module):   # λ预测网络
analytical_C()                # 解析解 C(z)

# ===================== 训练 =====================
class Trainer:                 # 训练器类
compute_loss()                # 损失计算
leave_one_out_cv()            # 留一交叉验证

# ===================== 可视化 =====================
plot_results()                # 结果绘图
```

---

## 10. 验证检查清单

- [ ] 数据加载成功（109样品，6-7剖面）
- [ ] 粘土矿物映射正确（Profile 3比例）
- [ ] 网络输出λ_H > λ_L（物理约束）
- [ ] Profile级别损失权重正常
- [ ] LOO-CV完成（6-7轮）
- [ ] 平均CV R²(L/H) > 0.5
- [ ] 所有图表正确生成
- [ ] 模型保存成功

---

## 11. 与v10/v9的关键差异

| 方面 | v10/v9 | v15 |
|------|--------|-----|
| **输出** | TotalREE (单值) | λ_L, λ_H (双参数) |
| **LREE/HREE** | 合并预测 | 分离建模（λ_H > λ_L） |
| **物理约束** | 无 | 解析PDE解 + λ_H > λ_L |
| **粘土矿物** | 无 | Profile 3比例映射 |
| **边界条件** | 软约束 | 解析解精确满足 |
| **可解释性** | 黑箱 | λ→k→物理意义 |
| **正则化** | 少 | Profile级别+L2 |

---

## 12. 预期结果

### 12.1 定量指标
- 训练集 R²(L/H) > 0.8
- LOO-CV R²(L/H) > 0.5
- 所有剖面满足 λ_H > λ_L

### 12.2 物理可解释性
- λ与气候（T, P, R）有合理相关性
- 粘土含量高的剖面应有更大的λ（吸附更强）
- λ_H > λ_L（HREE吸附更强）
