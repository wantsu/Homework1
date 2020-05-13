import torch.nn.functional as F
from project.dataloader import load_data
from project.model import MLP, CNN, RNN
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from project.test import Metrics

# Parameters
batch_size = 256
learning_rate = 0.01
weight_decay = 0.0001
num_epochs = 15
dropout = 0
batchnorm=True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# parameters
in_feature = 300
num_hidden_1 = 512
num_hidden_2 = 256
out_feature = 2



model_name = 'MLP'

## 初始化模型
if model_name == 'MLP':
    model = MLP(in_feature, out_feature)
elif model_name == 'CNN':
    model = CNN(in_feature, out_feature)
else:
    model = RNN(in_feature, out_feature)

model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
print('Model message:\n', model)


## Training
Lost = []
P, R, F1 = [], [], []
train_loader, test_loader = load_data(batch_size)
for epoch in range(num_epochs):
    model.train()
    for batch_idx, (features, targets) in enumerate(train_loader):
        features = features.to(device)
        targets = targets.to(device)

        ### FORWARD AND BACK PROP
        logits, probas = model(features)
        cost = F.cross_entropy(logits, targets.long())
        optimizer.zero_grad()

        cost.backward()
        ### UPDATE MODEL PARAMETERS
        nn.utils.clip_grad_norm_(model.parameters(), 10.)
        optimizer.step()
        ### LOGGING
        if not batch_idx % 50:
            print('Epoch: %03d/%03d | Batch %03d/%03d | Cost: %.4f'
                  % (epoch + 1, num_epochs, batch_idx,
                     len(train_loader), cost))

        Lost.append(cost.item())

    p, r, f1, report = Metrics(model, test_loader)  # precision, recall, f1, report
    P.append(p)
    R.append(r)
    F1.append(f1)
    print('Epoch:', (epoch + 1), '\n', report)

plt.subplot(2, 2, 1)  # 两行两列，plt.subplot('行','列','编号')
plt.plot([i + 1 for i in range(len(Lost))], Lost)
plt.ylabel('lost')
plt.subplot(2, 2, 2)  # 两行两列,这是第二个图
plt.plot([i + 1 for i in range(num_epochs)], P)
plt.ylabel('precision')
plt.subplot(2, 2, 3)  # 两行两列,这是第三个图
plt.plot([i + 1 for i in range(num_epochs)], R)
plt.ylabel('recall')
plt.subplot(2, 2, 4)  # 两行两列,这是第四个图
plt.plot([i + 1 for i in range(num_epochs)], F1)
plt.ylabel('f1')
plt.show()

# 保存模型
torch.save(model, model_name)
