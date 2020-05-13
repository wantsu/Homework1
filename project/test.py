import torch
from project.dataloader import load_data
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import os

# Metrics: precision, recall, f1, loss
def Metrics(model, data_loader):
    model.eval()
    labels, predicted = [], []
    with torch.no_grad():
        for features, targets in data_loader:
            features = features.to(device)
            targets = targets.to(device)
            _, probas = model(features)
            _, pred = torch.max(probas, 1)
            labels.append(targets)
            predicted.append(pred)

        labels = torch.cat(labels, dim=0)
        predicted = torch.cat(predicted, dim=0)
        report = classification_report(labels, predicted)
        p = precision_score(labels, predicted)  # precision
        r = recall_score(labels, predicted)  # recall
        f1 = f1_score(labels, predicted)  # f1
        return p, r, f1, report

if __name__ == '__main__':
    model_name = 'MLP'
    batch_size = 256
    if os.path.exists(model_name):
        model = torch.load(model_name)   # 加载模型
        _, test_loader = load_data(batch_size)
        p, r, f1, report = Metrics(model, test_loader)
        print('Test precisison:', p, 'recall:', r, 'f1:', f1)
        print('\n', report)
    else:
        print("Can't find model")