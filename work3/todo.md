# Work3 TODO

## 基于Mamba的光学器件正向/逆向设计

### 任务列表

- [x] 1. config.py - 配置文件（INPUT_DIM=6, OUTPUT_DIM=500, Mamba参数）
- [x] 2. utils.py - 工具函数
- [x] 3. data_loader.py - 数据加载（input3行列与work2相反，无需转置）
- [x] 4. models.py - Mamba正向网络 + MLP后向网络 + 串联网络
- [x] 5. train_forward.py - Mamba正向网络训练
- [x] 6. train_tandem.py - 串联网络训练
- [x] 7. evaluate.py - 评估与可视化
- [x] 8. main.py - 主程序入口
- [x] 9. requirements.txt
- [x] 10. 验证代码可运行（模型构建、数据加载、梯度回传均通过）

### Mamba 架构细节

- SelectiveSSM: A固定(log-spaced HiPPO初始化), B/C/dt输入依赖(选择性机制)
- MambaBlock: LayerNorm → Linear(D→2E) → 双分支(Conv1d+SiLU+SSM / SiLU门控) → 逐元素相乘 → Linear(E→D) → 残差
- 配置: d_model=64, d_state=16, d_conv=4, expand=2, n_layers=4
- 前向网络参数量: 167,924; 后向网络参数量: 690,822

### 数据集信息

- 样本数: 7814 (训练集 7032, 测试集 782)
- 输入: 6维结构参数
- 输出: 500维光谱（原始范围 [-31.7, 23.8]，MinMax归一化到[0,1]）
- 数据存储: 行=样本, 列=特征（与work2的input行列对调）

### 使用方式

```bash
python main.py            # 完整流程
python main.py --train    # 仅训练
python main.py --evaluate # 仅评估
python main.py --demo     # 逆向设计演示
```
