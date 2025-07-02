import torch
import os
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import PolynomialFeatures

from homework_datasets import CSVDataset
from homework_model_modification import LinearRegressionL1L2

import sys
sys.stdout.reconfigure(encoding='utf-8')

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def run_experiment(learning_rates, batch_sizes, optimizers, dataset):
    results = []
    
    for lr in learning_rates:
        for batch_size in batch_sizes:
            for opt_name in optimizers:
                logging.info(f"Тест: LR={lr}, Batch={batch_size}, Optimizer={opt_name}")
                
                model = LinearRegressionL1L2(dataset.X.shape[1])
                
                if opt_name == 'SGD':
                    optimizer = optim.SGD(model.parameters(), lr=lr)
                elif opt_name == 'Adam':
                    optimizer = optim.Adam(model.parameters(), lr=lr)
                elif opt_name == 'RMSprop':
                    optimizer = optim.RMSprop(model.parameters(), lr=lr)

                dataset_size = len(dataset)
                val_size = int(0.2 * dataset_size)
                train_size = dataset_size - val_size
                train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=batch_size)

                criterion = nn.MSELoss()
                train_losses = []
                val_losses = []

                for epoch in range(10):
                    model.train()
                    epoch_loss = 0
                    for X_batch, y_batch in train_loader:
                        optimizer.zero_grad()
                        output = model(X_batch).squeeze()
                        loss = criterion(output, y_batch.float())
                        loss.backward()
                        optimizer.step()
                        epoch_loss += loss.item()
                    train_losses.append(epoch_loss / len(train_loader))

                    model.eval()
                    with torch.no_grad():
                        val_loss = sum(criterion(model(X_batch).squeeze(), y_batch.float()).item() for X_batch, y_batch in val_loader) / len(val_loader)
                        val_losses.append(val_loss)

                model_dir = "models"
                os.makedirs(model_dir, exist_ok=True)
                model_path = os.path.join(model_dir, f"linear_regression_lr{lr}_bs{batch_size}_{opt_name}.pt")
                torch.save(model.state_dict(), model_path)
                logging.info(f"Модель сохранена в {model_path}")
                
                results.append({
                    'lr': lr,
                    'batch_size': batch_size,
                    'optimizer': opt_name,
                    'final_val_loss': val_losses[-1]
                })

    df = pd.DataFrame(results)
    df.sort_values('final_val_loss', inplace=True)
    logging.info("Лучшие параметры:\n" + str(df.head()))
    df.to_csv("plots/hyperparameter_results.csv", index=False)
    return df


def add_feature_engineering(df, numerical_columns):
    """Генерация дополнительных признаков."""
    logging.info("Добавление новых признаков (полиномиальные, взаимодействия, статистика)...")

    # Полиномиальные признаки
    poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
    poly_features = poly.fit_transform(df[numerical_columns])
    poly_feature_names = poly.get_feature_names_out(numerical_columns)
    poly_df = pd.DataFrame(poly_features, columns=poly_feature_names)

    # Статистические признаки
    poly_df['row_mean'] = poly_df.mean(axis=1)
    poly_df['row_std'] = poly_df.std(axis=1)

    df_new = pd.concat([df.drop(columns=numerical_columns), poly_df], axis=1)
    logging.info(f"Размерность данных после добавления признаков: {df_new.shape}")
    return df_new


def plot_hyperparameter_results(csv_path):
    df = pd.read_csv(csv_path)
    plt.figure(figsize=(10, 6))
    for optimizer in df['optimizer'].unique():
        sub = df[df['optimizer'] == optimizer]
        plt.plot(sub['lr'], sub['final_val_loss'], label=optimizer, marker='o')
    plt.xlabel("Learning Rate")
    plt.ylabel("Validation Loss")
    plt.title("График зависимости ошибки от гиперпараметров")
    plt.legend()
    plt.grid(True)
    plt.savefig("plots/hyperparameter_graph.png")
    plt.close()
