# REE-PINNs 增强方案

## 改进目标

在现有纯物理驱动的 PINNs 基础上，集成实测稀土数据、黏土矿物、基岩类型、pH 等地质化学数据，实现物理约束与数据驱动的双驱动建模，提升稀土分布预测精度。

---

## 1. 当前架构分析

### 现有架构（纯物理驱动）

```
输入: (t, z)                          时间 t + 深度 z
   │
   ▼
PINN 神经网络
   │  hidden_layers: [64, 64, 64, 64]
   │
   ▼
输出: (C_LREE, C_HREE)                轻稀土 + 重稀土浓度
   │
   ▼
物理方程损失: ∂C/∂t = D·∂²C/∂z² - u·∂C/∂z - k_ads·C + S
```

### 现有损失函数

| 损失项 | 权重 | 作用 |
|--------|------|------|
| `loss_physics` | 1.0 | PDE 方程约束 |
| `loss_fractionation` | 0.1 | 分馏机制约束 |
| `loss_bc` | 1.0 | 边界条件约束 |

---

## 2. 增强架构设计

### 目标架构（物理+数据双驱动）

```
输入: (t, z, clay, bedrock, pH, ...)
   │
   ├─ t: 时间坐标 [year]
   ├─ z: 深度坐标 [m]
   ├─ clay: 黏土矿物含量 [%]
   ├─ bedrock: 基岩类型 [类别编码]
   └─ pH: 酸碱度
   │
   ▼
增强版 PINN 神经网络
   │
   ├─ 物理分支: 学习 PDE 规律
   ├─ 数据分支: 学习实测数据特征
   └─ 特征融合: 综合输出
   │
   ▼
输出: (C_LREE, C_HREE)
   │
   ▼
多目标损失函数
   ├─ 物理方程损失
   ├─ 实测数据损失
   ├─ 黏土矿物损失
   ├─ pH 影响损失
   └─ 边界条件损失
```

---

## 3. 改进方案

### 3.1 扩展网络输入维度

**修改文件**: `ree_pinns_model.py`

**修改内容**: `REE_PINN` 类

```python
class REE_PINN(nn.Module):
    def __init__(self, input_dim=6, hidden_dims=[64, 64, 64, 64], output_dim=2):
        """
        增强版 PINN

        输入维度说明:
        - input_dim=2: 基础版 (t, z)
        - input_dim=3: (t, z, clay)
        - input_dim=5: (t, z, clay, bedrock_onehot, pH)
        - input_dim=6: 完整版
        """
        super(REE_PINN, self).__init__()
        self.input_dim = input_dim
        # ... 网络结构
```

**输入特征设计**:

| 索引 | 特征名 | 类型 | 说明 |
|------|--------|------|------|
| 0 | t | 连续 | 时间坐标 |
| 1 | z | 连续 | 深度坐标 |
| 2 | clay | 连续 | 黏土矿物含量 (0-1) |
| 3 | bedrock | 类别 | 基岩类型 (one-hot) |
| 4 | pH | 连续 | 酸碱度 |
| 5 | temperature | 连续 | 温度（可选） |

---

### 3.2 增加实测数据损失项

**修改位置**: `train_step()` 方法

```python
def train_step(self, n_physics, n_bc, T, h):
    # 1. 物理方程损失（保持不变）
    residual_LREE, residual_HREE = self.physics.compute_physics_residuals(
        self.pinn, t_pc, z_pc, additional_features
    )
    loss_physics = torch.mean(residual_LREE**2) + torch.mean(residual_HREE**2)

    # 2. 新增：实测稀土数据损失
    if self.obs_data is not None:
        t_obs, z_obs, C_LREE_obs, C_HREE_obs = self.obs_data
        C_LREE_pred, C_HREE_pred = self.pinn(t_obs, z_obs)
        loss_obs = torch.mean((C_LREE_pred - C_LREE_obs)**2) + \
                   torch.mean((C_HREE_pred - C_HREE_obs)**2)
    else:
        loss_obs = 0.0

    # 3. 新增：黏土矿物损失
    if self.clay_data is not None:
        loss_clay = self.physics.compute_clay_loss(self.pinn, self.clay_data)

    # 4. 新增：pH 影响损失
    if self.pH_data is not None:
        loss_pH = self.physics.compute_pH_loss(self.pinn, self.pH_data)

    # 5. 边界条件损失（保持不变）
    # ... existing bc loss code ...

    # 总损失
    loss_total = (self.lambda_dict['physics'] * loss_physics +
                  self.lambda_dict['obs'] * loss_obs +
                  self.lambda_dict['clay'] * loss_clay +
                  self.lambda_dict['pH'] * loss_pH +
                  self.lambda_dict['bc'] * loss_bc)
```

---

### 3.3 修改物理方程（考虑异质性）

**原方程**（假设均匀介质）:

$$\frac{\partial C}{\partial t} = D \cdot \frac{\partial^2 C}{\partial z^2} - u \cdot \frac{\partial C}{\partial z} - k_{ads} \cdot C + S$$

**改进方程**（考虑空间异质性）:

$$\frac{\partial C}{\partial t} = D(x) \cdot \frac{\partial^2 C}{\partial z^2} - u \cdot \frac{\partial C}{\partial z} - k_{ads}(x) \cdot C + S(x)$$

其中:

- $D(x) = D_0 \cdot f_{clay}(clay) \cdot f_{bedrock}(bedrock)$
- $k_{ads}(x) = k_0 \cdot g_{clay}(clay) \cdot g_{pH}(pH)$

**新增参数**:

```python
# REEPhysics.__init__()
self.D_clay_factor = 1.0      # 黏土对扩散系数的影响系数
self.k_ads_pH_factor = 1.0    # pH 对吸附系数的影响系数
```

---

### 3.4 增加辅助损失函数

**黏土矿物损失**:

```python
def compute_clay_loss(self, pinn, clay_data):
    """
    约束黏土矿物对吸附的影响
    假设: 黏土含量越高，吸附越强，LREE/HREE 比值越高
    """
    z_clay, clay_content = clay_data
    t_dummy = torch.zeros_like(z_clay)

    # 预测在黏土层的分馏效应
    C_LREE, C_HREE = pinn(t_dummy, z_clay, clay=clay_content)

    # 约束: 高黏土 → 高 LREE/HREE 比值
    ratio = C_LREE / (C_HREE + 1e-6)
    target_ratio = 1.0 + clay_content * self.clay_fractionation_factor
    loss = torch.mean((ratio - target_ratio)**2)

    return loss
```

**pH 影响损失**:

```python
def compute_pH_loss(self, pinn, pH_data):
    """
    约束 pH 对稀土活性的影响
    假设: pH 影响稀土的吸附解吸平衡
    """
    z_pH, pH_values = pH_data
    t_dummy = torch.zeros_like(z_pH)

    # 预测在不同 pH 下的浓度
    C_LREE, C_HREE = pinn(t_dummy, z_pH, pH=pH_values)

    # 约束: 简化的 pH-吸附关系
    # 实际关系需要根据地球化学模型确定
    loss = torch.tensor(0.0, requires_grad=True)

    return loss
```

---

## 4. 数据集成规范

### 4.1 数据格式要求

| 数据类型 | 格式 | 字段 | 说明 |
|---------|------|------|------|
| **实测稀土** | `(N, 4)` | `t, z, C_LREE, C_HREE` | N 个实测点 |
| **黏土矿物** | `(M, 2)` | `z, clay_content` | M 个深度点 |
| **基岩类型** | `(K, 2)` | `z, bedrock_id` | K 个深度点 |
| **pH 数据** | `(P, 2)` | `z, pH` | P 个深度点 |

### 4.2 数据加载接口

```python
class EnhancedSimulator(REESimulator):
    def __init__(self, params=None, obs_data=None, clay_data=None,
                 bedrock_data=None, pH_data=None):
        """
        增强版模拟器

        参数:
            params: 物理参数字典
            obs_data: 实测稀土数据, shape=(N, 4), columns=[t, z, C_LREE, C_HREE]
            clay_data: 黏土矿物数据, shape=(M, 2), columns=[z, clay_content]
            bedrock_data: 基岩类型数据, shape=(K, 2), columns=[z, bedrock_id]
            pH_data: pH 数据, shape=(P, 2), columns=[z, pH]
        """
        super().__init__(params)

        # 数据归一化
        self.obs_data = self._normalize_obs_data(obs_data)
        self.clay_data = self._normalize_clay_data(clay_data)
        self.bedrock_data = self._encode_bedrock_data(bedrock_data)
        self.pH_data = self._normalize_pH_data(pH_data)

        # 调整网络输入维度
        input_dim = self._compute_input_dim()
        self.pinn = Enhanced_PINN(input_dim=input_dim, ...)

        # 调整损失权重
        self._adjust_lambda_dict()
```

### 4.3 归一化方法

```python
def _normalize_obs_data(self, obs_data):
    """实测数据归一化"""
    if obs_data is None:
        return None

    t, z, C_LREE, C_HREE = obs_data.T

    # Min-Max 归一化到 [0, 1]
    t_norm = (t - self.params['T_min']) / (self.params['T_max'] - self.params['T_min'])
    z_norm = z / self.params['h']
    C_LREE_norm = C_LREE / self.params['C_LREE_max']
    C_HREE_norm = C_HREE / self.params['C_HREE_max']

    return torch.tensor([t_norm, z_norm, C_LREE_norm, C_HREE_norm]).T

def _normalize_clay_data(self, clay_data):
    """黏土数据归一化"""
    if clay_data is None:
        return None

    z, clay_content = clay_data.T
    z_norm = z / self.params['h']
    clay_norm = clay_content  # 假设已是 [0, 1] 范围

    return torch.tensor([z_norm, clay_norm]).T
```

---

## 5. 损失权重调整策略

### 5.1 默认权重配置

```python
lambda_dict = {
    'physics': 1.0,       # 物理方程约束
    'obs': 10.0,          # 实测数据约束（权重较高）
    'clay': 0.5,          # 黏土矿物约束
    'pH': 0.5,            # pH 影响约束
    'bc': 1.0,            # 边界条件约束
    'fractionation': 0.1  # 分馏机制约束
}
```

### 5.2 权重调整建议

| 场景 | 权重调整建议 |
|------|-------------|
| 实测数据质量高 | 提高 `obs` 权重 (10-100) |
| 物理模型确定性高 | 提高 `physics` 权重 |
| 辅助数据噪声大 | 降低对应损失权重 |
| 训练不稳定 | 降低总损失权重，统一调小 |

---

## 6. 实现优先级

### Phase 1: 基础增强（高优先级）
1. 扩展网络输入维度支持
2. 增加实测稀土数据损失
3. 数据加载和归一化接口

### Phase 2: 物理增强（中优先级）
4. 考虑黏土矿物影响的 PDE 修改
5. 黏土矿物损失函数
6. 基岩类型 one-hot 编码

### Phase 3: 高级特性（低优先级）
7. pH 影响建模
8. 自适应权重调整
9. 多任务学习框架

---

## 7. 注意事项

1. **数据对齐**: 所有额外数据需与 (t, z) 坐标对齐
2. **归一化**: 不同量纲数据需标准化处理
3. **权重调优**: 通过实验确定各损失项最优权重
4. **数据质量**: 实测数据越准确，预测效果越好
5. **插值处理**: 对于非网格数据，需要插值到训练点

---

## 8. 参考代码结构

```
ree_pinns/
├── ree_pinns_model.py          # 原始模型
├── enhanced_model.py           # 增强版模型（新建）
├── data_loader.py              # 数据加载模块（新建）
└── PINN_ENHANCEMENT_PLAN.md    # 本文档
```
