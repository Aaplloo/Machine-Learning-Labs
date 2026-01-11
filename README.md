# 机器学习核心算法底层复现 (Machine Learning Algorithms: Implemented from Scratch)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![NumPy](https://img.shields.io/badge/Library-NumPy-orange)
![Status](https://img.shields.io/badge/Status-Educational-green)

> **"What I cannot create, I do not understand."** — Richard Feynman

## 📖 项目介绍
本项目旨在**从零开始（From Scratch）**复现经典的机器学习算法。所有核心逻辑主要依赖 `NumPy` 进行矩阵运算，**拒绝直接调用 `sklearn` 等现成的高级封装接口**。

相比于完成课程作业，本项目的核心目标是**打开算法的“黑盒”**。通过手动推导并实现**反向传播（Backpropagation）**、**信息增益（Information Gain）**及**贝叶斯推断**等数学过程，深入理解统计学习与优化理论的底层机理。

## 🛠️ 技术栈
* **核心算法实现：** Python + NumPy (纯手写数学逻辑)
* **文本处理：** Jieba (用于中文分词)
* **可视化分析：** Matplotlib / Seaborn
* **数据处理：** Pandas

## 🧮 算法实现列表

| 实验项目 | 算法名称 | 数学核心 | 实现亮点 |
| :--- | :--- | :--- | :--- |
| **Lab 1** | **ID3 决策树** | 信息增益 (Information Gain) | • 实现了递归建树与多叉树结构<br>• 加入了**预剪枝 (Pre-pruning)** 策略 (最大深度控制) 防止过拟合 |
| **Lab 2** | **BP 神经网络** | 链式法则 & 梯度下降 | • **手动实现反向传播**：推导并编码了全连接层的梯度更新公式<br>• **数据增广**：实现了图像随机裁剪与旋转<br>• **底层解析**：直接解析 MNIST `idx3-ubyte` 二进制文件 |
| **Lab 3** | **朴素贝叶斯** | MAP 估计 & 贝叶斯公式 | • **对数概率 (Log-Probabilities)**：将连乘转化为对数求和，解决数值下溢问题<br>• **拉普拉斯平滑**：处理未登录词的零概率问题 |

## 🚀 核心技术细节

### 1. ID3 决策树 (Lab 1)
基于**信息增益**准则选择最优划分属性。
$$Gain(D, a) = Ent(D) - \sum_{v=1}^{V} \frac{|D^v|}{|D|} Ent(D^v)$$
* **递归构建：** 使用字典结构动态构建多叉树。
* **模型优化：** 引入 `max_depth` 超参数控制树的复杂度，提升泛化能力。

### 2. BP 神经网络 (Lab 2)
搭建了一个包含隐藏层的全连接网络 (784 -> 128 -> 10) 用于 MNIST 手写数字识别。
* **手动反向传播 (Manual Backpropagation)：**
    不依赖 Autograd 框架，手动计算损失函数关于权重 $W$ 和偏置 $b$ 的偏导数：
    ```python
    # 核心代码片段：链式法则实现
    delta2 = self.a2 - y_true
    dW2 = np.dot(self.a1.T, delta2) / batch_size
    delta1 = np.dot(delta2, self.W2.T) * self.sigmoid_deriv(self.a1)
    dW1 = np.dot(x.T, delta1) / batch_size
    ```
* **工程实现：** 实现了基于 `idx` 格式的文件读取器，不依赖 `torchvision` 等库加载数据。

### 3. 朴素贝叶斯文本分类 (Lab 3)
应用于新闻标题分类任务。为保证数值稳定性，在预测阶段使用对数似然形式：
$$\hat{y} = \arg\max_{c} (\log P(c) + \sum_{i=1}^{n} \log P(x_i | c))$$
* **平滑处理：** 应用 $\lambda=1$ 的拉普拉斯平滑，有效解决了测试集中生僻词导致的零概率问题。
* **指标评估：** 手写实现了 Precision, Recall, F1-score 的计算逻辑，详细分析各类别的分类性能。

## 📂 项目文件结构
```text
.
├── Lab1-ID3算法分类及剪枝/
│   ├── Lab1.py            # 决策树核心实现
│   └── result.csv         # 预测结果文件
├── Lab2-基于BP神经网络算法的手写数字识别/
│   ├── code.py            # BP网络与训练脚本
│   └── experiment_results/# 训练曲线与混淆矩阵图
└── Lab3-基于朴素贝叶斯的文本分类算法/
    ├── Lab3.py            # 朴素贝叶斯实现
    └── *_metrics.png      # 性能评估图表
