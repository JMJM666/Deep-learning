from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import pandas as pd
import copy
import torch

# 选择 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用设备：", device)

# 读取原始数据
data = pd.read_csv("covid.train.csv")
test_data = pd.read_csv("covid.test.csv")

# 随机取出 20% 作为验证集，剩余部分用于训练
valid_data = data.sample(n=int(len(data) * 0.2), random_state=42)
train_data = data.drop(index=valid_data.index)

print("训练集：", train_data.shape)
print("验证集：", valid_data.shape)
print("测试集：", test_data.shape)

# 要预测的目标：第 5 天阳性率
target = "tested_positive.4"

# 输入特征：去掉编号和目标列
x_train = train_data.drop(columns=["id", target])
x_valid = valid_data.drop(columns=["id", target])
x_test = test_data.drop(columns=["id"])

# 只根据训练集计算标准化参数
feature_mean = x_train.mean()
feature_std = x_train.std(ddof=0).replace(0, 1)

# 三份数据使用同一组参数
x_train = (x_train - feature_mean) / feature_std
x_valid = (x_valid - feature_mean) / feature_std
x_test = (x_test - feature_mean) / feature_std

print("标准化后训练特征：", x_train.shape)
print("训练特征前5列的均值：")
print(x_train.mean().head())

# 真实目标值
y_train = train_data[target]
y_valid = valid_data[target]

# 输入特征转成 float32 张量
x_train_tensor = torch.tensor(
    x_train.to_numpy(), dtype=torch.float32
)
x_valid_tensor = torch.tensor(
    x_valid.to_numpy(), dtype=torch.float32
)
x_test_tensor = torch.tensor(
    x_test.to_numpy(), dtype=torch.float32
)

# 目标值转成张量
y_train_tensor = torch.tensor(
    y_train.to_numpy(), dtype=torch.float32
)
y_valid_tensor = torch.tensor(
    y_valid.to_numpy(), dtype=torch.float32
)

# 将每条输入与对应的真实目标配对
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
valid_dataset = TensorDataset(x_valid_tensor, y_valid_tensor)

# 每批最多取 64 条数据
train_loader = DataLoader(
    train_dataset, batch_size=64, shuffle=True
)
valid_loader = DataLoader(
    valid_dataset, batch_size=64, shuffle=False
)

# 检查一批数据能否放到 GPU
batch_x, batch_y = next(iter(train_loader))
batch_x = batch_x.to(device)
batch_y = batch_y.to(device)

print("一批输入的形状：", batch_x.shape)
print("一批目标的形状：", batch_y.shape)
print("所在设备：", batch_x.device)


print("训练特征：", x_train.shape)
print("训练目标：", y_train.shape)
print("验证特征：", x_valid.shape)
print("验证目标：", y_valid.shape)
print("测试特征：", x_test.shape)

# 116 个输入特征 -> 64 -> 32 -> 1 个预测值
class RegressionModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.layers(x).squeeze(1)


# 创建模型，并放到 GPU
model = RegressionModel(
    input_dim=x_train_tensor.shape[1]
).to(device)

# 用刚才的一批数据检查输出
model.eval()
with torch.no_grad():
    predictions = model(batch_x)

print(model)
print("预测值的形状：", predictions.shape)
print("前5个预测值：", predictions[:5].cpu().numpy())

# MSE：预测值与真实值之间的均方误差
criterion = nn.MSELoss()

# 优化器负责更新网络参数
optimizer = torch.optim.Adam(
    model.parameters(), lr=0.001, weight_decay=0.0001
)

best_mse = float("inf")
best_state = None
wait = 0

for epoch in range(200):
    model.train()
    train_total = 0.0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        predictions = model(inputs)
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()

        train_total += loss.item() * len(inputs)

    # 验证阶段只评估，不更新参数
    model.eval()
    valid_total = 0.0
    with torch.no_grad():
        for inputs, targets in valid_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            loss = criterion(model(inputs), targets)
            valid_total += loss.item() * len(inputs)

    train_mse = train_total / len(train_dataset)
    valid_mse = valid_total / len(valid_dataset)

    if valid_mse < best_mse:
        best_mse = valid_mse
        best_state = copy.deepcopy(model.state_dict())
        wait = 0
    else:
        wait += 1

    if epoch == 0 or (epoch + 1) % 10 == 0:
        print(
            f"第 {epoch + 1} 轮："
            f"训练 MSE={train_mse:.4f}，验证 MSE={valid_mse:.4f}"
        )

    if wait >= 30:
        print("连续30轮没有改善，提前停止。")
        break

# 恢复验证表现最好的参数
model.load_state_dict(best_state)
model.eval()
print(f"最佳验证 MSE：{best_mse:.4f}")

from pathlib import Path

# 测试时保持原始行顺序
test_loader = DataLoader(
    TensorDataset(x_test_tensor),
    batch_size=256,
    shuffle=False
)

model.eval()
test_predictions = []

with torch.no_grad():
    for (inputs,) in test_loader:
        inputs = inputs.to(device)
        predictions = model(inputs)
        test_predictions.append(predictions.cpu())

# 合并所有批次的预测结果
test_predictions = torch.cat(test_predictions).numpy()

# 保留原始测试集的编号
submission = pd.DataFrame({
    "id": test_data["id"].to_numpy(),
    "tested_positive": test_predictions
})

# 保存到 hw1.py 所在目录
output_path = Path(__file__).resolve().parent / "pred.csv"
submission.to_csv(output_path, index=False)

print(submission.head())
print("预测结果形状：", submission.shape)
print("已保存到：", output_path)
