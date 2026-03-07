# Work2 TODO

## 基于LSTM的光学器件正向/逆向设计

### 任务列表

- [x] 1. config.py - 配置文件（INPUT_DIM=4, LSTM隐藏层[10,30,50,80]等）
- [x] 2. utils.py - 工具函数（set_seed, get_device, EarlyStopping）
- [x] 3. data_loader.py - 数据加载与预处理（适配4维输入，6129组数据）
- [x] 4. models.py - 模型定义（LSTM正向网络 + MLP后向网络 + 串联网络）
- [x] 5. train_forward.py - LSTM正向网络训练
- [x] 6. train_tandem.py - 串联网络训练（逆向设计）
- [x] 7. evaluate.py - 模型评估与可视化（含R²、参数对比图、CSV导出）
- [x] 8. main.py - 主程序入口
- [x] 9. requirements.txt - 依赖列表
- [x] 10. 验证代码可运行（模型构建、数据加载均通过测试）

### 相比work1的改进

- 正向网络改用LSTM架构（论文: 4层LSTM, hidden=[10,30,50,80]）
- 增加R²评估指标
- 增加逆向设计参数对比散点图
- 增加CSV结果导出功能
- 增加LayerNorm稳定LSTM训练
- LSTM权重使用正交初始化+遗忘门偏置初始化
- 批次大小从32增大到64（适配更大数据集6129组 vs 2911组）

### 数据集信息

- 样本数: 6129组
- 输入: 4维结构参数（P, Lx, Ly, h）
- 输出: 200维吸收光谱
- 训练集: 5516, 测试集: 613

### 使用方式

```bash
# 完整流程
python main.py

# 仅训练
python main.py --train

# 仅评估
python main.py --evaluate

# 逆向设计演示
python main.py --demo
```
