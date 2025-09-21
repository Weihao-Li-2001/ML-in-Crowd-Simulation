# This is a file that assemble model generation selection validation

from sklearn.model_selection import KFold
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

def build_mlp(input_dim, output_dim, architecture):
    """
    Build a simple Multi-Layer Perceptron (MLP) using PyTorch Sequential API.

    Parameters
    ----------
    input_dim : int, default=21
        Number of input features (dimensionality of X).
    output_dim : int, default=1
        Number of outputs (e.g. 1 for regression, number of classes for classification).
    architecture : tuple of int, default=(3,)
        Sizes of hidden layers. For example:
        - (32,)        → 1 hidden layer with 32 units
        - (64, 32, 16) → 3 hidden layers with 64, 32, and 16 units

    Returns
    -------
    torch.nn.Sequential
        A PyTorch Sequential model representing the MLP.
    """
    layers = []
    in_features = input_dim

    for hidden_size in architecture:
        layers.append(nn.Linear(in_features, hidden_size))
        layers.append(nn.ReLU())
        in_features = hidden_size

    layers.pop()
    layers.append(nn.Linear(in_features, output_dim))       
    return nn.Sequential(*layers)


# def train_one_fold(model, optimizer, loss_fn, X_train, y_train, X_val, y_val, num_epochs):
#     for epoch in range(num_epochs):
#         model.train()
#         optimizer.zero_grad()
#         y_pred = model(X_train)
#         train_loss = loss_fn(y_train, y_pred)
#         train_loss.backward()
#         optimizer.step()
#     model.eval()
#     with torch.no_grad():
#         y_val_pred = model(X_val)
#         val_loss = loss_fn(y_val, y_val_pred)
#     return train_loss.item(), val_loss.item()


def evaluate_with_bootstrap_cv(model_builder, df, features, target_col="v",
                               n_bootstrap=50, n_splits=5, num_epochs=100, lr=0.001,
                               loss_fn=None):
    """
    Evaluate a PyTorch model with Bootstrap + K-Fold Cross Validation.

    Parameters
    ----------
    model_arch : tuple
        Architecture of the MLP, e.g. (32, 16) means two hidden layers with 32 and 16 units.
    df : pandas.DataFrame
        Input dataset containing features and target.
    features : list of str
        List of feature column names.
    target_col : str, default="v"
        Column name of the target variable in df.
    n_bootstrap : int, default=50
        Number of bootstrap resamples.
    n_splits : int, default=5
        Number of folds for K-Fold cross-validation.
    num_epochs : int, default=100
        Number of training epochs per fold.
    lr : float, default=0.001
        Learning rate for Adam optimizer.

    Future Work
    -------
    This is a giant monster, we could seperate it into several parts that we could use in the future
    
    Returns
    -------
    dict
        Dictionary containing mean and std of train/validation losses across
        bootstrap resamples:
        {
            "train_mean": float,
            "train_std": float,
            "val_mean": float,
            "val_std": float
        }
    """
    # prepare datasets
    df = df.copy()
    X = df[features].values          # features
    y = df[target_col].values        # target

    # prepare training
    if loss_fn is None:
        loss_fn = nn.MSELoss()
    bootstrap_train_losses = []
    bootstrap_val_losses = []
    
    for i in range(n_bootstrap):
        # resampling to make bootstrap datasets
        X_resampled, y_resampled = resample(X, y, replace=True, n_samples=len(X))
        kf = KFold(n_splits=n_splits, shuffle=True) # for each bootstrap should carry a unique KFold
        fold_train_losses, fold_val_losses = [], []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_resampled)):
            # train/val split
            X_train, X_val = X_resampled[train_idx], X_resampled[val_idx]
            y_train, y_val = y_resampled[train_idx], y_resampled[val_idx]

            # normalize
            scaler = StandardScaler()
            X_train_norm = scaler.fit_transform(X_train)
            X_val_norm = scaler.transform(X_val)
            
            # tensors
            X_train_tensor = torch.tensor(X_train_norm, dtype=torch.float32)
            y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
            X_val_tensor = torch.tensor(X_val_norm, dtype=torch.float32)
            y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

            # model + optimizer
            model = mlp_builder(input_dim=X_train.shape[1], output_dim=1)
            optimizer = optim.Adam(model.parameters(), lr=lr)

            # training loop
            model.train()
            # 先不用dataloader试试看
            for epoch in range(num_epochs):
                optimizer.zero_grad()
                y_train_pred = model(X_train_tensor)
                train_loss = loss_fn(y_train_tensor, y_train_pred)
                train_loss.backward()
                optimizer.step()

            # record train loss
            fold_train_losses.append(train_loss.item())

            # validation
            model.eval()
            with torch.no_grad():
                y_val_pred = model(X_val_tensor)
                val_loss = loss_fn(y_val_tensor, y_val_pred)
                fold_val_losses.append(val_loss.item())
                
        bootstrap_train_losses.append(np.mean(fold_train_losses))
        bootstrap_val_losses.append(np.mean(fold_val_losses))

    return {
        "train_mean": np.mean(bootstrap_train_losses),
        "train_std":  np.std(bootstrap_train_losses),
        "val_mean":   np.mean(bootstrap_val_losses),
        "val_std":    np.std(bootstrap_val_losses)
    }