import os
import sys
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
import logging

from homework_datasets import CSVDataset
from homework_experiments import run_experiment, plot_hyperparameter_results, add_feature_engineering
from homework_model_modification import LinearRegressionL1L2, LogisticRegressionCustom
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

sys.stdout.reconfigure(encoding='utf-8')

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def run_car_regression():
    logging.info("=== Запуск экспериментов с регрессией (Car dataset) ===")

    csv_path = "data/car_data.csv"
    target_column = "Selling_Price"
    categorical_columns = ["Fuel_Type", "Seller_Type", "Transmission"]
    numerical_columns = ["Year", "Present_Price", "Kms_Driven", "Owner"]

    dataset = CSVDataset(
        file_path=csv_path,
        target_column=target_column,
        categorical_columns=categorical_columns,
        numerical_columns=numerical_columns
    )

    learning_rates = [0.001, 0.01, 0.1]
    batch_sizes = [16, 32]
    optimizers = ["SGD", "Adam", "RMSprop"]

    results_df = run_experiment(learning_rates, batch_sizes, optimizers, dataset)
    plot_hyperparameter_results("plots/hyperparameter_results.csv")

    raw_df = pd.read_csv(csv_path)
    fe_df = add_feature_engineering(raw_df, numerical_columns=numerical_columns)
    fe_df.to_csv("data/engineered_car_data.csv", index=False)

def run_titanic_classification():
    logging.info("=== Запуск бинарной классификации Titanic ===")

    csv_path = "data/Titanic-Dataset.csv"
    target_column = "Survived"
    categorical_columns = ["Sex", "Embarked"]
    numerical_columns = ["Pclass", "Age", "SibSp", "Parch", "Fare"]

    dataset = CSVDataset(
        file_path=csv_path,
        target_column=target_column,
        categorical_columns=categorical_columns,
        numerical_columns=numerical_columns
    )

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    model = LogisticRegressionCustom(dataset.X.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCEWithLogitsLoss()

    epochs = 20
    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            logits = model(X_batch)  # Логиты без сигмоида
            loss = criterion(logits, y_batch.float())
            loss.backward()
            optimizer.step()
        logging.info(f"Epoch {epoch+1}/{epochs} completed.")

    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            output = model(X_batch).squeeze()
            preds = (output >= 0.5).int()
            all_preds.extend(preds.tolist())
            all_targets.extend(y_batch.tolist())

    acc = accuracy_score(all_targets, all_preds)
    report = classification_report(all_targets, all_preds)
    logging.info(f"Accuracy: {acc:.4f}")
    logging.info(f"Classification report:\n{report}")

    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix - Titanic")
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/titanic_confusion_matrix.png")
    plt.close()
    logging.info("Confusion matrix saved to plots/titanic_confusion_matrix.png")

if __name__ == "__main__":
    run_car_regression()
    run_titanic_classification()