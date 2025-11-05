from src.data_and_preprocessors.preprocessors import build_transformer_for_regression
import torch 
from torch.utils.data import DataLoader, TensorDataset

def mlp_prepare_data(X_train, X_test, y_train, y_test, numerical_cols, categorical_cols):
    sc_1 = build_transformer_for_regression(numerical_cols.tolist(), categorical_cols.tolist())
    sc_1.fit(X_train)

    X_train_scaled = sc_1.transform(X_train)
    X_test_scaled = sc_1.transform(X_test)

    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.FloatTensor(y_train.values).reshape(-1, 1)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_test_tensor = torch.FloatTensor(y_test.values).reshape(-1, 1)

    train_data = TensorDataset(X_train_tensor, y_train_tensor)
    test_data = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=True)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    weight = (y_train == 0).sum() / (y_train == 1).sum()
    input_dim = X_train_scaled.shape[1]

    return train_loader, test_loader, weight, input_dim