import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import logging
import sys

sys.stdout.reconfigure(encoding='utf-8')

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class LinearRegressionL1L2(nn.Module):
    """Линейная регрессия с поддержкой L1 и L2 регуляризации и Early Stopping."""
    def __init__(self, input_dim, l1_lambda=0.0, l2_lambda=0.0):
        super(LinearRegressionL1L2, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda

    def forward(self, x):
        return self.linear(x)

    def compute_loss(self, y_pred, y_true):
        mse_loss = F.mse_loss(y_pred, y_true)
        l1_reg = self.l1_lambda * torch.norm(self.linear.weight, 1)
        l2_reg = self.l2_lambda * torch.norm(self.linear.weight, 2)**2
        return mse_loss + l1_reg + l2_reg


def train_linear_model(model, optimizer, X_train, y_train, X_val, y_val, 
                       num_epochs=1000, patience=10):
    """Обучение модели линейной регрессии с early stopping."""
    best_loss = float('inf')
    epochs_no_improve = 0
    best_model = None

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = model.compute_loss(outputs, y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = model.compute_loss(val_outputs, y_val)

        logging.info(f"Epoch {epoch+1}: Train loss = {loss.item():.4f}, Val loss = {val_loss.item():.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            best_model = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            logging.info("Early stopping triggered.")
            break

    model.load_state_dict(best_model)
    return model


class MulticlassLogisticRegression(nn.Module):
    """Логистическая регрессия для многоклассовой классификации."""
    def __init__(self, input_dim, num_classes):
        super(MulticlassLogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)
    
    
class LogisticRegressionCustom(nn.Module):
    """
    Бинарная логистическая регрессия без сигмоиды (логиты).
    """
    def __init__(self, input_dim):
        super(LogisticRegressionCustom, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x).squeeze()


def evaluate_classification(model, X, y_true):
    """Оценка метрик классификации."""
    model.eval()
    with torch.no_grad():
        logits = model(X)
        y_pred = torch.argmax(logits, dim=1).cpu().numpy()
        y_true_np = y_true.cpu().numpy()
        
        precision = precision_score(y_true_np, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true_np, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true_np, y_pred, average='weighted')

        # Для ROC AUC нужен one-hot
        y_true_oh = F.one_hot(torch.tensor(y_true_np), num_classes=logits.shape[1]).numpy()
        y_proba = F.softmax(logits, dim=1).cpu().numpy()
        roc_auc = roc_auc_score(y_true_oh, y_proba, average='weighted', multi_class='ovr')

        logging.info(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")
        return precision, recall, f1, roc_auc


def plot_confusion_matrix(model, X, y_true, class_names, save_path=None):
    """Визуализация confusion matrix."""
    model.eval()
    with torch.no_grad():
        logits = model(X)
        y_pred = torch.argmax(logits, dim=1).cpu().numpy()
        y_true_np = y_true.cpu().numpy()

    cm = confusion_matrix(y_true_np, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    if save_path:
        plt.savefig(save_path)
    plt.close()
