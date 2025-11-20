import os
import jieba
import math
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict, Counter

plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False

# 数据加载
def load_data(file_name):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, file_name)
    data = []

    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue  # 跳过空行

                # 尝试多种分隔方式
                title, label = None, None
                for sep in ['\t', '  ']:
                    parts = line.split(sep)
                    if len(parts) >= 2:
                        label = parts[-1].strip()
                        title = sep.join(parts[:-1]).strip()
                        if title and label:
                            break
                if not label:
                    parts = [p for p in line.split(' ') if p.strip()]
                    if len(parts) >= 2:
                        label = parts[-1].strip()
                        title = ' '.join(parts[:-1]).strip()

                if title and label:
                    data.append((title, label))

        print(f"加载{file_name}：有效数据{len(data)}条")
    except FileNotFoundError:
        print(f"错误：未找到文件{file_name}（路径：{file_path}）")
        return []
    except Exception as e:
        print(f"加载{file_name}失败：{str(e)}")
        return []
    return data

# 文本预处理
def preprocess_text(text):
    words = jieba.lcut(text)
    filtered_words = [word for word in words if len(word) > 1 and not (word.isdigit() or word in ',.!?;:"\'()[]{}')]
    return filtered_words

# 模型训练
def train_naive_bayes(train_processed):
    label_count = Counter()
    word_label_count = defaultdict(Counter)
    label_total_words = defaultdict(int)
    vocab = set()

    total_samples = len(train_processed)
    print(f"\n开始训练模型：共{total_samples}条数据")
    for idx, (words, label) in enumerate(train_processed, 1):
        if idx % 10000 == 0 or idx == total_samples:
            print(f"训练进度：{idx}/{total_samples}（{idx/total_samples*100:.1f}%）")
        
        if not words:
            continue
        
        label_count[label] += 1
        for word in words:
            word_label_count[label][word] += 1
            label_total_words[label] += 1
            vocab.add(word)

    all_labels = list(label_count.keys())
    N = sum(label_count.values())
    K = len(all_labels)
    V = len(vocab)
    print(f"训练完成：{K}个类别，词库大小{V}，有效样本{N}条")
    return all_labels, label_count, word_label_count, label_total_words, vocab, N, K

# 预测与评估
def predict_single(words, model_params):
    if not model_params:
        return None
    all_labels, label_count, word_label_count, label_total_words, vocab, N, K = model_params
    V = len(vocab)
    max_log_prob = -float('inf')
    best_label = all_labels[0]

    for label in all_labels:
        count_Ck = label_total_words[label]
        if count_Ck == 0:
            log_prior = math.log(1 / (N + K))
            log_likelihood = 0.0
        else:
            N_k = label_count[label]
            log_prior = math.log((N_k + 1) / (N + K))
            log_likelihood = 0.0
            for word in words:
                count_word_Ck = word_label_count[label].get(word, 0)
                log_p_xi_Ck = math.log((count_word_Ck + 1) / (count_Ck + V))
                log_likelihood += log_p_xi_Ck
        
        if log_prior + log_likelihood > max_log_prob:
            max_log_prob = log_prior + log_likelihood
            best_label = label
    return best_label

def batch_predict(data_processed, model_params, data_name):
    y_true, y_pred = [], []
    total_samples = len(data_processed)
    print(f"\n开始{data_name}预测：共{total_samples}条数据")
    for idx, (words, label) in enumerate(data_processed, 1):
        if idx % 10000 == 0 or idx == total_samples:
            print(f"{data_name}进度：{idx}/{total_samples}（{idx/total_samples*100:.1f}%）")
        
        y_true.append(label)
        if not words:
            pred_label = max(model_params[1], key=model_params[1].get)
        else:
            pred_label = predict_single(words, model_params)
        y_pred.append(pred_label)
    return y_true, y_pred

def calculate_metrics(y_true, y_pred, all_labels, data_name):
    # 计算基础指标
    correct = sum(t == p for t, p in zip(y_true, y_pred))
    accuracy = correct / len(y_true) if len(y_true) > 0 else 0.0

    # 计算混淆矩阵相关指标
    tp, fp, fn = defaultdict(int), defaultdict(int), defaultdict(int)
    for t, p in zip(y_true, y_pred):
        if t == p:
            tp[t] += 1
        else:
            fp[p] += 1
            fn[t] += 1

    # 计算各类别指标
    class_metrics = {"precision": {}, "recall": {}, "f1": {}}
    for label in all_labels:
        p = tp[label] / (tp[label] + fp[label]) if (tp[label] + fp[label]) > 0 else 0.0
        r = tp[label] / (tp[label] + fn[label]) if (tp[label] + fn[label]) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        class_metrics["precision"][label] = p
        class_metrics["recall"][label] = r
        class_metrics["f1"][label] = f1

    # 计算宏平均指标
    macro_precision = sum(class_metrics["precision"].values()) / len(all_labels)
    macro_recall = sum(class_metrics["recall"].values()) / len(all_labels)
    macro_f1 = sum(class_metrics["f1"].values()) / len(all_labels)

    # 打印文字结果
    print(f"\n{data_name}指标：")
    print(f"准确率：{accuracy:.4f} | 宏平均F1：{macro_f1:.4f}")
    print(f"类别详情：")
    print(f"{'类别':<10} {'精确率':<5} {'召回率':<5} {'F1':<5}")
    for label in all_labels:
        print(f"{label:<10} {class_metrics['precision'][label]:.4f}   {class_metrics['recall'][label]:.4f}   {class_metrics['f1'][label]:.4f}")

    # 绘制图表
    plot_metrics(class_metrics, all_labels, data_name, accuracy, macro_precision, macro_recall, macro_f1)

    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "class_metrics": class_metrics
    }

# 图表绘制功能
def plot_metrics(class_metrics, all_labels, data_name, accuracy, macro_precision, macro_recall, macro_f1):
    # 各类别 P / R / F1对比图
    x = np.arange(len(all_labels))  # 类别索引
    width = 0.25  # 柱状图宽度

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))  # 1 行 2 列图表

    # 绘制类别性能对比图
    ax1.bar(x - width, [class_metrics["precision"][label] for label in all_labels], width, label='精确率')
    ax1.bar(x, [class_metrics["recall"][label] for label in all_labels], width, label='召回率')
    ax1.bar(x + width, [class_metrics["f1"][label] for label in all_labels], width, label='F1值')

    ax1.set_title(f'{data_name}各类别性能指标对比', fontsize=12)
    ax1.set_xlabel('类别', fontsize=10)
    ax1.set_ylabel('指标值', fontsize=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(all_labels, rotation=45, ha='right')
    ax1.set_ylim(0, 1.0)
    ax1.legend()

    # 整体指标柱状图
    metrics = ['准确率', '宏平均精确率', '宏平均召回率', '宏平均F1']
    values = [accuracy, macro_precision, macro_recall, macro_f1]
    
    ax2.bar(metrics, values, color=['#5DA5DA', '#FAA43A', '#60BD68', '#F17CB0'])
    ax2.set_title(f'{data_name}整体性能指标', fontsize=12)
    ax2.set_ylabel('指标值', fontsize=10)
    ax2.set_ylim(0, 1.0)
    
    # 在柱状图上标注具体数值
    for i, v in enumerate(values):
        ax2.text(i, v + 0.02, f'{v:.4f}', ha='center', fontsize=9)

    # 调整布局避免重叠
    plt.tight_layout()
    # 保存图表
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(current_dir, f'{data_name}_metrics.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"{data_name}指标图表已保存至：{save_path}")
    # 显示图表
    plt.show()

# 结果保存
def save_test_results(test_data, y_true, y_pred):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(current_dir, "test_pred_results.txt")
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("新闻标题\t真实类别\t预测类别\n")
        for (title, _), t, p in zip(test_data, y_true, y_pred):
            f.write(f"{title}\t{t}\t{p}\n")
    print(f"\n预测结果已保存至：{save_path}")

def main():
    print("="*66)
    print("朴素贝叶斯文本分类")
    print("="*66)

    # 加载数据
    train_data = load_data("train.txt")
    dev_data = load_data("dev.txt")
    test_data = load_data("test.txt")

    # 预处理
    train_processed = [(preprocess_text(title), label) for title, label in train_data]
    dev_processed = [(preprocess_text(title), label) for title, label in dev_data] if dev_data else []
    test_processed = [(preprocess_text(title), label) for title, label in test_data] if test_data else []
    print(f"预处理完成：训练集{len(train_processed)}条，验证集{len(dev_processed)}条，测试集{len(test_processed)}条")

    # 训练模型
    model_params = train_naive_bayes(train_processed)
    all_labels = model_params[0]

    # 验证集评估
    if dev_processed:
        y_true_dev, y_pred_dev = batch_predict(dev_processed, model_params, "验证集")
        calculate_metrics(y_true_dev, y_pred_dev, all_labels, "验证集")

    # 测试集预测与评估
    if test_processed:
        y_true_test, y_pred_test = batch_predict(test_processed, model_params, "测试集")
        calculate_metrics(y_true_test, y_pred_test, all_labels, "测试集")
        save_test_results(test_data, y_true_test, y_pred_test)

if __name__ == "__main__":
    main()