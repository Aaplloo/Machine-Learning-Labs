import pandas as pd
import numpy as np
from collections import Counter

def load_datasets(train_path='train.csv', predict_path='predict.csv'): # 读取数据
    try:
        train_df = pd.read_csv(train_path)
        predict_df = pd.read_csv(predict_path)
        return train_df, predict_df
    except FileNotFoundError as e:
        print(f"数据集文件未找到：{e}，请检查文件路径是否正确")
        return None, None

def calc_entropy( data ): # 计算信息熵

    label_counts = Counter(data.iloc[:, -1])
    total_samples = len(data)
    entropy = 0.0
    
    for count in label_counts.values():
        prob = count / total_samples
        if prob > 0: # 避免基数为 0
            entropy -= prob * np.log2( prob )
    return entropy

def calc_info_gain(data, feature_index): # 计算信息增益

    base_entropy = calc_entropy( data )
    feature_values = data.iloc[:, feature_index].unique()
    weighted_entropy = 0.0
    
    for value in feature_values:
        # 按当前属性值划分数据集
        subset = data[data.iloc[:, feature_index] == value]
        weight = len( subset ) / len( data )
        weighted_entropy += weight * calc_entropy( subset )
    
    info_gain = base_entropy - weighted_entropy
    return info_gain

def select_best_feature(data): # 选择信息增益最大的属性作为当前决策树节点的划分属性
    feature_count = len( data.columns ) - 1
    max_gain = -1
    best_feature_idx = -1
    
    for idx in range(feature_count):
        current_gain = calc_info_gain(data, idx)
        if current_gain > max_gain:
            max_gain = current_gain
            best_feature_idx = idx
    return best_feature_idx

def split_dataset(data, feature_index, feature_value): # 移除已使用的属性列，划分数据集

    subset = data[data.iloc[:, feature_index] == feature_value].copy()
    subset = subset.drop(subset.columns[feature_index], axis = 1)  # 移除已划分属性
    return subset

def majority_vote( labels ): # 叶节点采用多数投票法
    label_count = Counter( labels )
    return max(label_count.items(), key=lambda x: x[1])[0]

def build_id3_tree(data, feature_names, max_depth = None, current_depth = 0): # 构造 ID3 决策树

    labels = data.iloc[:, -1]
    
    # 所有样本类别相同，直接返回该类别
    if len(labels.unique()) == 1:
        return labels.iloc[0]
    
    # 无属性可划分，返回多数类别
    if len(data.columns) == 1:
        return majority_vote( labels )
    
    # 预剪枝：达到最大深度，返回多数类别
    if max_depth is not None and current_depth >= max_depth:
        return majority_vote( labels )
    
    # 选择最优划分属性并构建子树
    best_feature_idx = select_best_feature( data )
    best_feature_name = feature_names[best_feature_idx]
    tree = {best_feature_name: {}}
    
    # 更新剩余属性名称列表
    remaining_feature_names = feature_names[:best_feature_idx] + feature_names[best_feature_idx+1:]
    best_feature_values = data.iloc[:, best_feature_idx].unique()
    
    # 递归构建子树
    for value in best_feature_values:
        subset = split_dataset(data, best_feature_idx, value)
        tree[best_feature_name][value] = build_id3_tree(
            subset, remaining_feature_names, max_depth, current_depth + 1
        )
    return tree

def predict_single_sample(tree, feature_names, sample): # 递归遍历决策树预测单个样本
    root_feature = list(tree.keys())[0]
    root_subtree = tree[root_feature]
    feature_idx = feature_names.index(root_feature)
    sample_value = sample.iloc[feature_idx]
    
    if sample_value in root_subtree:
        child_node = root_subtree[sample_value]
        if isinstance(child_node, dict):
            return predict_single_sample(child_node, feature_names, sample)
        else:
            return child_node
    else:
        return majority_vote(pd.read_csv('train.csv').iloc[:, -1])

def predict_batch_samples(tree, feature_names, predict_df): # 预测所有样本
    predictions = []
    for idx in range(len(predict_df)):
        sample = predict_df.iloc[idx]
        pred_label = predict_single_sample(tree, feature_names, sample)
        predictions.append(pred_label)
    
    # 生成含预测结果的数据集
    result_df = predict_df.copy()
    result_df['weather'] = predictions
    return result_df

def print_tree_pretty(tree, indent="", is_last = True):
    if not isinstance(tree, dict):  # 叶节点
        print(f"{indent}{'└── ' if is_last else '├── '}类别：{tree}")
        return
    # 根节点 / 中间节点
    root_feature = list(tree.keys())[0]
    print(f"{indent}{'└── ' if is_last else '├── '}属性：{root_feature}")

    feature_values = list(tree[root_feature].items())
    for idx, (value, child_tree) in enumerate(feature_values):
        is_last_child = (idx == len(feature_values) - 1)
        child_indent = indent + ("    " if is_last else "│   ")
        print(f"{child_indent}{'└── ' if is_last_child else '├── '}取值：{value} →")
        
        print_tree_pretty(child_tree, child_indent, is_last_child)

def main():

    train_df, predict_df = load_datasets()
    if train_df is None or predict_df is None:
        return
    
    feature_names = [col for col in train_df.columns if col != 'weather']
    train_data = train_df[feature_names + ['weather']]
    
    id3_tree = build_id3_tree(train_data, feature_names, max_depth=3)
    print("决策树构建完成: ")
    print_tree_pretty( id3_tree )
    
    predict_result = predict_batch_samples(id3_tree, feature_names, predict_df)
    print("预测完成, 预测结果前5行: ")
    print( predict_result.head() )
    
    # 保存预测结果到result.csv
    predict_result.to_csv('result.csv', index = False, encoding = 'utf-8')
    print("\n预测结果已保存至 result.csv 文件")

if __name__ == "__main__":
    main()