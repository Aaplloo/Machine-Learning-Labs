import os
import gzip
import matplotlib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 读取 MNIST 的 idx3-ubyte 和 idx1-ubyte 格式数据
def read_idx3_ubyte(file_path):
    if file_path.endswith('.gz'):
        with gzip.open(file_path, 'rb') as f:
            data = f.read()
    else:
        with open(file_path, 'rb') as f:
            data = f.read()
    
    # 解析头部信息（大端序）：魔数(4 bytes) ： 图像数量(4 bytes) ： 行数(4 bytes) ： 列数(4 bytes)
    magic_num = int.from_bytes(data[0:4], byteorder='big')
    num_imgs = int.from_bytes(data[4:8], byteorder='big')
    rows = int.from_bytes(data[8:12], byteorder='big')
    cols = int.from_bytes(data[12:16], byteorder='big')
    
    # 验证文件格式（idx3-ubyte 魔数为 2051）
    assert magic_num == 2051, f"错误：{file_path}不是 idx3-ubyte 格式文件，魔数应为 2051"
    
    # 解析图像数据（每个像素为 8 位无符号整数，取值 0 - 255）
    imgs = np.frombuffer(data[16:], dtype=np.uint8)
    # 重塑为[样本数, 行数, 列数]的三维数组（适配28 × 28灰度图）
    imgs = imgs.reshape(num_imgs, rows, cols)
    return imgs

def read_idx1_ubyte(file_path):
    if file_path.endswith('.gz'):
        with gzip.open(file_path, 'rb') as f:
            data = f.read()
    else:
        with open(file_path, 'rb') as f:
            data = f.read()
    
    # 解析头部信息：魔数(4 bytes) ： 标签数量(4 bytes)
    magic_num = int.from_bytes(data[0:4], byteorder='big')
    num_labels = int.from_bytes(data[4:8], byteorder='big')
    
    # 验证文件格式（idx1-ubyte 魔数为 2049）
    assert magic_num == 2049, f"错误：{file_path}不是 idx1-ubyte 格式文件，魔数应为 2049"
    
    # 解析标签数据（每个标签为0-9的整数）
    labels = np.frombuffer(data[8:], dtype = np.uint8)
    return labels

# 数据预处理（归一化、展平、独热编码）
def preprocess_data(imgs, labels, is_train=True, val_split=5000):
    """
    imgs: 原始图像数组（[样本数, 28, 28]）
    labels: 原始标签数组（[样本数, 1]）
    is_train: 是否为训练集（True则划分验证集，False不划分）
    val_split: 验证集样本数（默认5000，匹配实验任务书中的训练集55000、验证集5000）
    """
    # 归一化：将像素值从 [0, 255] 缩放到 [0, 1]，避免数值过大导致梯度爆炸
    imgs_norm = imgs.astype(np.float32) / 255.0
    
    # 图像展平：将 28 × 28 的二维图像转为 784 的一维向量（适配 BP 神经网络输入层）
    imgs_flat = imgs_norm.reshape(-1, 28*28)  # 形状变为[样本数, 784]
    
    # 标签独热编码：多分类任务，将标签转为独热向量
    def one_hot_encode(labels, num_classes=10):
        num_samples = len(labels)
        one_hot = np.zeros((num_samples, num_classes), dtype=np.float32)
        one_hot[np.arange(num_samples), labels] = 1.0
        return one_hot
    labels_onehot = one_hot_encode(labels)
    
    # 训练集划分验证集（按实验任务书要求，训练集 55000、验证集 5000）
    if is_train:
        # 训练集：前 N - val_split 个样本， 验证集：后 val_split 个样本
        train_imgs = imgs_flat[:-val_split]
        train_labels = labels_onehot[:-val_split]
        val_imgs = imgs_flat[-val_split:]
        val_labels = labels_onehot[-val_split:]
        return train_imgs, train_labels, val_imgs, val_labels
    else:
        # 测试集不划分
        return imgs_flat, labels_onehot

# 数据增广，对训练集图像进行随机裁剪、旋转等来扩充样本数量和多样性， 提升模型泛化能力
def augment_train_data(imgs, labels, augment_times=1):
    """
    imgs: 预处理后的训练集图像（[55000, 784]）
    labels: 预处理后的训练集标签（[55000, 10]）
    augment_times: 每张原图生成的增广样本数（default：1，生成后训练集变为 110000样本）
    """
    augmented_imgs = []
    augmented_labels = []
    num_samples = len(imgs)
    
    for i in range(num_samples):
        # 将展平的图像恢复为 28 × 28 的二维格式
        img_flat = imgs[i]
        img_2d = img_flat.reshape(28, 28)
        
        # 生成augment_times个增广样本
        for _ in range(augment_times):
            # 随机裁剪：先在图像四周填充 2 个像素（变为32 × 32），再随机裁剪回 28 × 28
            pad_img = np.pad(img_2d, pad_width=2, mode='constant', constant_values=0)
            crop_x = np.random.randint(0, pad_img.shape[0] - 28)
            crop_y = np.random.randint(0, pad_img.shape[1] - 28)
            cropped_img = pad_img[crop_x:crop_x+28, crop_y:crop_y+28]
            
            # 随机旋转：-10° ~ 10°之间随机旋转（简化实现，基于矩阵旋转）
            angle = np.random.randint(-10, 11)
            rad = np.radians(angle)
            cos_rad = np.cos(rad)
            sin_rad = np.sin(rad)
            center = (14, 14)
            rotated_img = np.zeros_like(cropped_img)
            for x in range(28):
                for y in range(28):
                    x_centered = x - center[0]
                    y_centered = y - center[1]
                    x_rot = int(x_centered * cos_rad - y_centered * sin_rad + center[0])
                    y_rot = int(x_centered * sin_rad + y_centered * cos_rad + center[1])
                    if 0 <= x_rot < 28 and 0 <= y_rot < 28:
                        rotated_img[x, y] = cropped_img[x_rot, y_rot]
            
            # 将增广后的图像重新展平，并添加到列表中
            augmented_imgs.append(rotated_img.flatten())
            augmented_labels.append(labels[i])
    
    # 转换为 numpy 数组，并与原始数据合并
    augmented_imgs = np.array(augmented_imgs)
    augmented_labels = np.array(augmented_labels)
    new_train_imgs = np.concatenate([imgs, augmented_imgs], axis=0)
    new_train_labels = np.concatenate([labels, augmented_labels], axis=0)
    
    # 打乱合并后的数据，避免顺序影响训练
    shuffle_idx = np.random.permutation(len(new_train_imgs))
    new_train_imgs = new_train_imgs[shuffle_idx]
    new_train_labels = new_train_labels[shuffle_idx]
    
    return new_train_imgs, new_train_labels

# BP神经网络类（纯 Numpy 实现）
class BPNeuralNetwork:
    # 初始化神经网络
    def __init__(self, input_dim=784, hidden_dim=128, output_dim=10, lr=0.01):
        """
        input_dim: 输入层维度（MNIST图像展平后为784）
        hidden_dim: 隐藏层神经元数量（可调整，default: 128）
        output_dim: 输出层维度（MNIST为 10 分类，故为 10）
        lr: 学习率（控制参数更新步长，default: 0.01）
        """
        # 初始化权重与偏置（采用Xavier初始化，避免初始值过大/过小导致梯度问题）
        self.W1 = np.random.randn(input_dim, hidden_dim) / np.sqrt(input_dim / 2)  # 输入层 → 隐藏层权重
        self.b1 = np.zeros((1, hidden_dim))  # 隐藏层偏置（初始为 0）
        self.W2 = np.random.randn(hidden_dim, output_dim) / np.sqrt(hidden_dim / 2)  # 隐藏层 → 输出层权重
        self.b2 = np.zeros((1, output_dim))  # 输出层偏置（初始为 0）
        
        # 超参数与训练记录
        self.lr = lr  # 学习率
        self.loss_history = []  # 记录每轮训练的平均损失
        self.val_acc_history = []  # 记录每轮训练的验证集准确率

    # 激活函数：sigmoid
    def sigmoid(self, x):
        x = np.clip(x, -500, 500) # 加入微小值避免数值溢出
        return 1 / (1 + np.exp(-x))
    
    # sigmoid 导数（反向传播计算梯度用）
    def sigmoid_deriv(self, x):
        return x * (1 - x)
    
    # 激活函数：softmax（输出层用，将输出转为概率分布）
    def softmax(self, x):
        # 减最大值避免数值溢出
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    # 损失函数：交叉熵（多分类任务常用，比均方误差更适合）
    def cross_entropy_loss(self, y_pred, y_true):
        eps = 1e-10  # 避免log(0)导致的数值错误
        return -np.mean(np.sum(y_true * np.log(y_pred + eps), axis=1))
    
    # 前向传播：计算各层输出
    def forward(self, x):
        # 隐藏层：z1 = 输入 × 权重1 + 偏置1 → a1 = sigmoid( z1 )
        self.z1 = np.dot(x, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        
        # 输出层：z2 = 隐藏层输出 × 权重2 + 偏置2 → a2 = softmax( z2 )
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.softmax(self.z2)
        
        return self.a2  # 返回输出层概率预测值
    
    # 反向传播：计算梯度并更新权重/偏置
    def backward(self, x, y_true):
        batch_size = x.shape[0]  # 批量大小（用于平均梯度）
        
        # 计算输出层误差（交叉熵 + softmax的导数简化结果：δ2 = 预测值 - 真实值）
        delta2 = self.a2 - y_true
        
        # 计算隐藏层误差（δ1 = δ2 × 权重2^T × sigmoid 导数( a1 )）
        delta1 = np.dot(delta2, self.W2.T) * self.sigmoid_deriv(self.a1)
        
        # 计算权重与偏置的梯度（平均梯度，避免批量大小影响）
        dW2 = np.dot(self.a1.T, delta2) / batch_size  # 隐藏层 → 输出层权重梯度
        db2 = np.sum(delta2, axis=0, keepdims=True) / batch_size  # 输出层偏置梯度
        dW1 = np.dot(x.T, delta1) / batch_size  # 输入层 → 隐藏层权重梯度
        db1 = np.sum(delta1, axis=0, keepdims=True) / batch_size  # 隐藏层偏置梯度
        
        # 梯度下降更新参数（W = W - 学习率 × 梯度，b = b - 学习率 × 梯度）
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
    
    # 模型训练：批量梯度下降
    def train(self, x_train, y_train, x_val, y_val, epochs=20, batch_size=64):
        """
        x_train: 训练集图像（[样本数, 784]）
        y_train: 训练集标签（[样本数, 10]，独热编码）
        x_val: 验证集图像（[5000, 784]）
        y_val: 验证集标签（[5000, 10]，独热编码）
        epochs: 训练轮数（默认20，可调整）
        batch_size: 批量大小（默认64，可调整）
        """
        num_samples = x_train.shape[0]
        num_batches = num_samples // batch_size  # 每轮训练的批次数
        
        print(f"开始训练：共{epochs}轮，每轮{num_batches}批，每批{batch_size}样本")
        
        for epoch in range(epochs):
            # 打乱训练数据，避免数据顺序影响训练效果
            shuffle_idx = np.random.permutation(num_samples)
            x_train_shuffle = x_train[shuffle_idx]
            y_train_shuffle = y_train[shuffle_idx]
            
            # 批量训练
            total_loss = 0.0  # 累计每轮的总损失
            for i in range(num_batches):
                # 取当前批次的数据
                batch_x = x_train_shuffle[i*batch_size : (i+1)*batch_size]
                batch_y = y_train_shuffle[i*batch_size : (i+1)*batch_size]
                
                # 前向传播：计算预测值
                y_pred = self.forward(batch_x)
                # 计算当前批次的交叉熵损失
                batch_loss = self.cross_entropy_loss(y_pred, batch_y)
                total_loss += batch_loss
                # 反向传播：更新参数
                self.backward(batch_x, batch_y)
            
            # 计算每轮的平均损失并记录
            avg_loss = total_loss / num_batches
            self.loss_history.append(avg_loss)
            
            # 验证集评估：计算准确率
            val_pred = self.predict(x_val)  # 预测验证集类别
            val_true = np.argmax(y_val, axis=1)  # 验证集真实类别，从独热编码转回
            val_acc = np.mean(val_pred == val_true)  # 验证集准确率
            self.val_acc_history.append(val_acc)
            
            # 打印每轮训练信息
            print(f"第{epoch+1:2d}/{epochs}轮 | 平均损失：{avg_loss:.4f} | 验证集准确率：{val_acc:.4f}")
    
    # 模型预测：输入图像，输出类别（0 - 9）
    def predict(self, x):
        y_pred_prob = self.forward(x)  # 前向传播得到概率分布
        return np.argmax(y_pred_prob, axis=1)  # 取概率最大的类别作为预测结果

# 模型评估与可视化函数，计算测试集准确率、生成分类报告、绘制混淆矩阵和训练曲线
def evaluate_model(model, x_test, y_test, save_fig=True, fig_path="."):
    """
    model: 训练好的BP神经网络模型
    x_test: 测试集图像（[10000, 784]）
    y_test: 测试集标签（[10000, 10]，独热编码）
    save_fig: 是否保存图表（默认True）
    fig_path: 图表保存路径（默认当前目录）
    """
    # 测试集预测
    test_pred = model.predict(x_test)
    test_true = np.argmax(y_test, axis=1)
    
    # 计算测试集准确率
    test_acc = np.mean(test_pred == test_true)
    print(f"\n================ 模型评估结果 ================")
    print(f"测试集准确率：{test_acc:.4f}")
    
    # 生成分类报告（精确率、召回率、F1值）
    print("\n分类报告（精确率/召回率/F1值）：")
    print(classification_report(
        test_true, test_pred,
        target_names=[str(i) for i in range(10)],
        digits=4
    ))
    
    # 绘制混淆矩阵
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(test_true, test_pred)
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=[str(i) for i in range(10)],
        yticklabels=[str(i) for i in range(10)]
    )
    plt.xlabel('预测标签', fontsize=12)
    plt.ylabel('真实标签', fontsize=12)
    plt.title('BP神经网络手写数字识别混淆矩阵', fontsize=14)
    if save_fig:
        plt.savefig(os.path.join(fig_path, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 绘制训练曲线（损失 + 验证准确率）
    plt.figure(figsize=(12, 4))
    
    # 子图1：训练损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(model.loss_history)+1), model.loss_history, 'b-', linewidth=2)
    plt.xlabel('训练轮数（Epoch）', fontsize=12)
    plt.ylabel('交叉熵损失', fontsize=12)
    plt.title('训练损失变化曲线', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # 子图2：验证准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(model.val_acc_history)+1), model.val_acc_history, 'r-', linewidth=2)
    plt.xlabel('训练轮数（Epoch）', fontsize=12)
    plt.ylabel('验证集准确率', fontsize=12)
    plt.title('验证集准确率变化曲线', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_fig:
        plt.savefig(os.path.join(fig_path, 'train_history.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n图表已保存至：{fig_path}")
    return test_acc

def main():
    
    dataset_dir = "."  # 数据集存放目录（默认 code.py 源文件夹）
    epochs = 20  # 训练轮数
    batch_size = 64  # 批量大小
    hidden_dim = 128  # 隐藏层神经元数
    lr = 0.01  # 学习率
    use_augmentation = True  # 是否使用数据增广
    
    # 读取数据集
    print("正在读取数据集...")
    train_imgs = read_idx3_ubyte(os.path.join(dataset_dir, "train-images-idx3-ubyte.gz"))
    train_labels = read_idx1_ubyte(os.path.join(dataset_dir, "train-labels-idx1-ubyte.gz"))
    test_imgs = read_idx3_ubyte(os.path.join(dataset_dir, "t10k-images-idx3-ubyte.gz"))
    test_labels = read_idx1_ubyte(os.path.join(dataset_dir, "t10k-labels-idx1-ubyte.gz"))
    
    # 验证数据集规模
    assert len(train_imgs) == 60000 and len(train_labels) == 60000, "训练集样本数错误（应为60000）"
    assert len(test_imgs) == 10000 and len(test_labels) == 10000, "测试集样本数错误（应为10000）"
    print(f"数据集读取完成：训练集共{len(train_imgs)}个样本，测试集共{len(test_imgs)}个样本")
    
    # 数据预处理
    print("正在进行数据预处理...")
    # 训练集预处理（划分55000训练、5000验证）
    train_imgs_prep, train_labels_prep, val_imgs_prep, val_labels_prep = preprocess_data(
        train_imgs, train_labels, is_train=True
    )
    # 测试集预处理
    test_imgs_prep, test_labels_prep = preprocess_data(
        test_imgs, test_labels, is_train=False
    )
    print(f"预处理完成：\n- 训练集：{train_imgs_prep.shape} | 验证集：{val_imgs_prep.shape} | 测试集：{test_imgs_prep.shape}")
    
    # 数据增广
    if use_augmentation:
        print("正在进行数据增广...")
        train_imgs_aug, train_labels_aug = augment_train_data(
            train_imgs_prep, train_labels_prep, augment_times=1
        )
        print(f"数据增广完成：训练集变为{train_imgs_aug.shape}样本")
    else: # 不进行数据增广
        train_imgs_aug, train_labels_aug = train_imgs_prep, train_labels_prep
    
    # 初始化并训练BP神经网络
    print("\n正在初始化BP神经网络...")
    bp_model = BPNeuralNetwork(
        input_dim=784,
        hidden_dim=hidden_dim,
        output_dim=10,
        lr=lr
    )
    
    print("\n开始训练BP神经网络...")
    bp_model.train(
        x_train=train_imgs_aug,
        y_train=train_labels_aug,
        x_val=val_imgs_prep,
        y_val=val_labels_prep,
        epochs=epochs,
        batch_size=batch_size
    )
    
    # 模型评估
    print("\n正在评估模型性能...")
    test_acc = evaluate_model(
        model=bp_model,
        x_test=test_imgs_prep,
        y_test=test_labels_prep,
        save_fig=True,
        fig_path="./experiment_results"
    )
    
    print(f"\n实验完成！最终测试集准确率：{test_acc:.4f}")

if __name__ == "__main__":
    
    os.makedirs("./experiment_results", exist_ok=True)
    main()