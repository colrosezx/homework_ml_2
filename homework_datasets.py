import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
import os
import logging
import sys

sys.stdout.reconfigure(encoding='utf-8')

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CSVDataset(Dataset):
    """
    Кастомный PyTorch Dataset для загрузки и предобработки CSV данных.

    Поддерживает числовые, бинарные и категориальные признаки.
    """
    def __init__(self, file_path, target_column, categorical_columns=None, numerical_columns=None):
        """
        :param file_path: путь к CSV-файлу
        :param target_column: название колонки-цели
        :param categorical_columns: список категориальных признаков
        :param numerical_columns: список числовых признаков
        """
        assert os.path.exists(file_path), f"Файл {file_path} не найден."
        
        self.data = pd.read_csv(file_path, encoding='utf-8')
        self.target_column = target_column

        self.categorical_columns = categorical_columns if categorical_columns else []
        self.numerical_columns = numerical_columns if numerical_columns else []

        self.features = self.data.drop(columns=[target_column])
        self.labels = self.data[target_column]

        self.preprocessor = self._create_preprocessor()
        self.X = self.preprocessor.fit_transform(self.features)
        self.y = self._encode_target(self.labels)

        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.long)

        logging.info(f"Загружен датасет из {file_path}: {self.X.shape[0]} образцов, {self.X.shape[1]} признаков.")

    def _create_preprocessor(self):
        """Создание пайплайна предобработки."""
        transformers = []

        if self.categorical_columns:
            transformers.append(('cat', OneHotEncoder(handle_unknown='ignore'), self.categorical_columns))
        if self.numerical_columns:
            transformers.append(('num', StandardScaler(), self.numerical_columns))

        return ColumnTransformer(transformers=transformers, remainder='drop')

    def _encode_target(self, y):
        """Кодирование целевой переменной."""
        if y.dtype == 'object' or y.dtype.name == 'category':
            encoder = LabelEncoder()
            return encoder.fit_transform(y)
        return y.to_numpy()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]