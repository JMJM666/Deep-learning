from pathlib import Path

import torch


# 当前 Python 文件所在的文件夹
project_dir = Path(__file__).resolve().parent

# 数据所在的文件夹
data_dir = (
    project_dir
    / "ml2022spring-hw2"
    / "libriphone"
    / "libriphone"
)

# 训练数据列表
train_split_path = data_dir / "train_split.txt"

# 音位标签文件
train_labels_path = data_dir / "train_labels.txt"


# 读取第一条语音的名称
with open(train_split_path, "r", encoding="utf-8") as file:
    utterance_id = file.readline().strip()

print("第一条语音的名称：", utterance_id)


# 根据名称找到对应的特征文件
feature_path = data_dir / "feat" / "train" / f"{utterance_id}.pt"

# 读取特征
features = torch.load(feature_path, map_location="cpu", weights_only=True)


# 在标签文件中找到这条语音对应的标签
with open(train_labels_path, "r", encoding="utf-8") as file:
    for line in file:
        parts = line.strip().split()

        if parts[0] == utterance_id:
            labels = [int(label) for label in parts[1:]]
            break
    else:
        raise ValueError(f"没有找到语音 {utterance_id} 对应的标签")

print("标签数量：", len(labels))
print("前 10 个标签：", labels[:10])
print("特征帧数和标签数是否相等：", len(features) == len(labels))

# 选择一帧，取它前后各一帧
frame_index = 100

previous_frame = features[frame_index - 1]
current_frame = features[frame_index]
next_frame = features[frame_index + 1]

# 沿着特征维度拼接
combined = torch.cat(
    [previous_frame, current_frame, next_frame],
    dim=0,
)

def get_context(features, frame_index):
    """取当前帧及其前后各一帧，并拼接成一个向量。"""

    # max 防止前一帧的编号小于 0
    previous_index = max(frame_index - 1, 0)

    # min 防止后一帧的编号超过最后一帧
    next_index = min(frame_index + 1, len(features) - 1)

    previous_frame = features[previous_index]
    current_frame = features[frame_index]
    next_frame = features[next_index]

    return torch.cat(
        [previous_frame, current_frame, next_frame],
        dim=0,
    )


# 分别测试开头、中间和结尾的帧
for index in [0, 100, len(features) - 1]:
    context = get_context(features, index)
    print(f"第 {index} 帧：拼接后形状 {context.shape}，标签 {labels[index]}")

from torch.utils.data import Dataset


class PhoneDataset(Dataset):
    def __init__(self, features, labels):
        # 创建数据集时，保存特征和标签
        self.features = features
        self.labels = labels

    def __len__(self):
        # 数据集里有多少条样本
        return len(self.features)

    def __getitem__(self, index):
        # 根据编号，取出一条样本和对应标签
        context = get_context(self.features, index)
        label = self.labels[index]

        return context, label


# 用当前这段语音的数据创建数据集
dataset = PhoneDataset(features, labels)

print("\n--- Dataset 测试 ---")
print("数据集样本数：", len(dataset))

sample, target = dataset[100]

print("第 100 条样本的形状：", sample.shape)
print("第 100 条样本的标签：", target)

from torch.utils.data import DataLoader


# 把数据集包装成数据加载器，每次取 8 条样本
data_loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=False,
)

# 从数据加载器中取出第一批数据
batch_features, batch_labels = next(iter(data_loader))

print("\n--- DataLoader 测试 ---")
print("一批输入的形状：", batch_features.shape)
print("一批标签的形状：", batch_labels.shape)
print("这一批的标签：", batch_labels)

from torch import nn

class PhoneClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(117, 256),
            nn.ReLU(),
            nn.Linear(256, 41),
        )

    def forward(self, x):
        return self.network(x)


# 创建模型
model = PhoneClassifier()

# 让一批特征通过模型
outputs = model(batch_features)

print("\n--- 模型测试 ---")
print("模型输入形状：", batch_features.shape)
print("模型输出形状：", outputs.shape)
print("第一条样本的 41 个类别分数：")
print(outputs[0])

# 选出每条样本分数最高的类别
predictions = outputs.argmax(dim=1)

print("模型预测类别：", predictions)
print("实际类别：", batch_labels)

# 定义损失函数：衡量模型预测和正确答案之间的差距
criterion = nn.CrossEntropyLoss()

# 定义优化器：根据计算出的梯度调整模型参数
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 计算更新前的损失
loss = criterion(outputs, batch_labels)
print("\n--- 模型学习一次 ---")
print("更新前的损失：", loss.item())

# 清除旧梯度
optimizer.zero_grad()

# 反向传播，计算每个参数的梯度
loss.backward()

# 根据梯度更新模型参数
optimizer.step()

# 再次预测，观察更新后的损失
new_outputs = model(batch_features)
new_loss = criterion(new_outputs, batch_labels)

print("更新后的损失：", new_loss.item())

# 重新创建数据加载器，训练时打乱样本顺序
train_loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
)

print("\n--- 训练整段语音 ---")

# 重复学习数据 5 轮
for epoch in range(5):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for batch_features, batch_labels in train_loader:
        # 1. 预测
        outputs = model(batch_features)

        # 2. 计算损失
        loss = criterion(outputs, batch_labels)

        # 3. 更新参数
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 4. 累计损失和预测正确的数量
        batch_size = batch_labels.size(0)
        total_loss += loss.item() * batch_size

        predictions = outputs.argmax(dim=1)
        correct += (predictions == batch_labels).sum().item()
        total += batch_size

    average_loss = total_loss / total
    accuracy = correct / total

    print(
        f"第 {epoch + 1} 轮 | "
        f"平均损失：{average_loss:.4f} | "
        f"训练准确率：{accuracy:.2%}"
    )
