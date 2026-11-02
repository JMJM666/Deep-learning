from pathlib import Path
import random


# 找到数据目录
project_dir = Path(__file__).resolve().parent
data_dir = (
    project_dir
    / "ml2022spring-hw2"
    / "libriphone"
    / "libriphone"
)

# 读取所有训练语音的名称
train_split_path = data_dir / "train_split.txt"

with open(train_split_path, "r", encoding="utf-8") as file:
    utterance_ids = [line.strip() for line in file if line.strip()]

print("语音总数：", len(utterance_ids))
print("打乱前的前 3 条：", utterance_ids[:3])

# 固定随机种子，并打乱语音顺序
random.seed(1337)
random.shuffle(utterance_ids)

# 前 80% 用于训练，剩下 20% 用于验证
split_index = int(len(utterance_ids) * 0.8)

train_ids = utterance_ids[:split_index]
val_ids = utterance_ids[split_index:]

print("\n--- 数据划分 ---")
print("训练集语音数：", len(train_ids))
print("验证集语音数：", len(val_ids))
print("训练集前 3 条：", train_ids[:3])
print("验证集前 3 条：", val_ids[:3])

# 检查同一段语音是否同时出现在两个集合里
overlap = set(train_ids) & set(val_ids)
print("两个集合共有的语音数：", len(overlap))

# 读取所有语音的标签
labels_path = data_dir / "train_labels.txt"

label_dict = {}

with open(labels_path, "r", encoding="utf-8") as file:
    for line in file:
        parts = line.strip().split()

        # 第一个内容是语音名称
        utterance_id = parts[0]

        # 后面的内容是每一帧对应的类别编号
        labels = [int(label) for label in parts[1:]]

        # 用语音名称作为键，标签列表作为值
        label_dict[utterance_id] = labels

print("\n--- 标签读取 ---")
print("有标签的语音数量：", len(label_dict))

# 查看训练集第一段语音的标签
example_id = train_ids[0]
example_labels = label_dict[example_id]

print("示例语音名称：", example_id)
print("这段语音的标签数量：", len(example_labels))
print("前 10 个标签：", example_labels[:10])

# 检查语音列表里的名称是否都能找到标签
missing_ids = [
    utterance_id
    for utterance_id in utterance_ids
    if utterance_id not in label_dict
]

print("找不到标签的语音数量：", len(missing_ids))