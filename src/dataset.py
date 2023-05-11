# Some code taken from saint: https://github.com/somepago/saint
# Modified by Jialiang Wang
# Copyright (c) 2023-Current Jialiang Wang <jilwang804@gmail.com>
# License: TBD

import openml
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder


class TabDataset(Dataset):

    def __init__(self, X, y, cat_idx, con_idx, continuous_mean_std=None):

        cat_idx = list(cat_idx)
        con_idx = list(con_idx)
        X_mask = X['mask']
        X = X['data']

        self.X1 = torch.from_numpy(X[:, cat_idx].copy()).long()                 # categorical columns
        self.X2 = torch.from_numpy(X[:, con_idx].copy()).float()                # numerical columns
        self.X1_mask = torch.from_numpy(X_mask[:, cat_idx].copy()).long()       # categorical columns
        self.X2_mask = torch.from_numpy(X_mask[:, con_idx].copy()).long()       # numerical columns
        self.y = torch.from_numpy(y['instance'].copy()).long()
        self.cls = torch.zeros_like(self.y, dtype=torch.int)
        self.cls_mask = torch.ones_like(self.y, dtype=torch.int)

        if continuous_mean_std is not None:
            mean, std = continuous_mean_std
            self.X2 = (self.X2 - mean) / std

    def __len__(self):
        return len(self.y)

    # X_cat, X_con, y, X_cat_mask, X_con_mask
    def __getitem__(self, idx):
        return torch.cat((self.cls[idx], self.X1[idx])), self.X2[idx], self.y[idx], torch.cat(
            (self.cls_mask[idx], self.X1_mask[idx])), self.X2_mask[idx]


# Download specified dataset from openml and preprocess it
def data_prep_openml(ds_id, seed=None):

    if seed is not None:
        np.random.seed(seed)
    dataset = openml.datasets.get_dataset(ds_id)
    target = dataset.default_target_attribute
    if ds_id == 4535:
        target = 'V42'
    X, y, categorical_indicator, attribute_names = dataset.get_data(dataset_format="dataframe",
                                                                    target=target)

    na_entry = 'MissingValue'
    if ds_id == 4535:
        na_entry = ' Not in universe'
    elif ds_id == 44234:
        categorical_indicator = [False, True, True, True, True, False, True, True, True, False, True, False, False, False, False, True]
        na_entry = 'unknown'
    y = y.astype('category')

    cat_idx = list(np.where(np.array(categorical_indicator) == True)[0])
    con_idx = list(set(range(len(X.columns))) - set(cat_idx))
    categorical_columns = X.columns[cat_idx].tolist()
    continuous_columns = X.columns[con_idx].tolist()

    for col in categorical_columns:
        X[col] = X[col].astype("object")

    X["Set"] = np.random.choice(["train", "test"], p=[0.7, 0.3], size=(X.shape[0],))
    train_indices = X[X.Set == "train"].index
    test_indices = X[X.Set == "test"].index
    X = X.drop(columns=['Set'])

    temp = X.fillna('MissingValue')
    temp = temp.replace([na_entry], 'MissingValue')
    nan_mask = temp.ne('MissingValue').astype(int)

    cat_dims = []
    for col in categorical_columns:
        X[col] = X[col].fillna('MissingValue')
        X[col] = X[col].replace([na_entry], 'MissingValue')
        l_enc = LabelEncoder()
        X[col] = l_enc.fit_transform(X[col].values)
        cat_dims.append(len(l_enc.classes_))
    for col in continuous_columns:
        X.fillna(X.loc[train_indices, col].mean(), inplace=True)

    y = y.values
    l_enc = LabelEncoder()
    y = l_enc.fit_transform(y)
    n_class = len(l_enc.classes_)
    X_train, y_train = data_split(X, y, nan_mask, train_indices)
    X_test, y_test = data_split(X, y, nan_mask, test_indices)

    train_mean = np.array(X.iloc[train_indices, con_idx], dtype=np.float32).mean(0)
    train_std = np.array(X.iloc[train_indices, con_idx], dtype=np.float32).std(0)
    train_std = np.where(train_std < 1e-6, 1e-6, train_std)

    return cat_dims, cat_idx, con_idx, X_train, y_train, X_test, y_test, train_mean, train_std, n_class


# Bagging the dataset into bags of fixed size
def data_split(X, y, nan_mask, indices):
    x_d = {
        'data': X.values[indices],
        'mask': nan_mask.values[indices]
    }

    if x_d['data'].shape != x_d['mask'].shape:
        raise 'Shape of data not same as that of nan mask!'

    y_d = {
        'instance': y[indices].reshape(-1, 1)
    }
    return x_d, y_d

