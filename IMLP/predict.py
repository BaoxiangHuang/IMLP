import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import os
import orjson
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Gaussian Radial Basis Function Layer
class GaussianRBF(nn.Module):
    def __init__(self, input_dim):
        super(GaussianRBF, self).__init__()
        self.b = nn.Parameter(torch.randn(input_dim))  # Learnable bias

    def forward(self, x):
        # Calculate activation function f(X) = exp(-π * (X - b)²)
        pi = torch.pi
        diff = x - self.b
        return torch.exp(-pi * (diff ** 2))

# MLP Model with RBF Layers
class MLPE(nn.Module):
    def __init__(self, input_size):
        super(MLPE, self).__init__()
        self.embedding_lat = nn.Embedding(90, 16)
        self.embedding_lon = nn.Embedding(360, 16)
        self.embedding_sst = nn.Embedding(64, 16)
        self.embedding_date = nn.Embedding(8192, 16)

        self.mlp1 = nn.Sequential(
            nn.Linear(input_size, 64),
            GaussianRBF(64),
            nn.Linear(64, 128),
            nn.Dropout(0.1),
            GaussianRBF(128)
        )

        self.mlp2 = nn.Sequential(
            nn.Linear(128 + input_size, 64),  # Concatenate original input with MLP1 output
            nn.Dropout(0.1),
            GaussianRBF(64),
            nn.Linear(64, 300)
        )

    def forward(self, x):
        x0 = torch.cat((self.embedding_lat(torch.trunc(x[:, 0]).long()), torch.frac(x[:, 0]).unsqueeze(1)), 1)
        x1 = torch.cat((self.embedding_lon(torch.trunc(x[:, 1]).long()), torch.frac(x[:, 1]).unsqueeze(1)), 1)
        x2 = self.embedding_date(x[:, 2].long())
        x3 = torch.cat((self.embedding_sst(torch.trunc(x[:, 3]).long()), torch.frac(x[:, 3]).unsqueeze(1)), 1)
        embedded_x = torch.cat((x0, x1, x2, x3, x[:, -1].unsqueeze(1)), 1)
        mlp1_output = self.mlp1(embedded_x.float())
        combined_x = torch.cat((embedded_x, mlp1_output), 1).float()
        return self.mlp2(combined_x)

# Dataset Processing
class DatasetProcess:
    def __init__(self):
        self.file_list = os.listdir('pre_data')
        self.save_path = 'pre_data.json'
        self.data_path = 'pre_data/'
        self.values = None
        self.data = None
        self.dataset()

    def load(self, path):
        return pd.read_excel(os.path.join(self.data_path, path))

    def dataset(self):
        data = None
        for file in self.file_list:
            contents = self.load(file)
            self.values = contents.keys() if self.values is None else self.values

            chla = np.array(contents['Chla'])
            temp_data = contents.values
            data_index = np.argwhere(chla != '--')
            useful_data = temp_data[data_index.reshape(-1), :]
            data = useful_data if data is None else np.concatenate((data, useful_data), axis=0)

        self.data = data

    def save(self):
        data = {'value': self.values.values.tolist(), 'data': self.data.tolist()}
        with open(os.path.join(self.data_path, self.save_path), 'w') as f:
            f.write(orjson.dumps(data).decode())
        print('----Save Finish----')

# Custom Dataset
class ChlDataset(Dataset):
    def __init__(self):
        with open('data/data.json', 'r') as f:
            data0 = orjson.loads(f.read())
            train_data = np.array(data0['train']).astype(float)

        self.mean = np.mean(train_data[:, 3:-6], axis=0)
        self.std = np.std(train_data[:, 3:-6], axis=0)

        with open('pre_data/pre_data.json', 'r') as f:
            data = orjson.loads(f.read())
            pre_data = np.array(data['data']).astype(float)
        self.data = pre_data

        self.feature_index = [0, 1, 2, 3, 4]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index, self.feature_index]
        return torch.from_numpy(x), x  # Return feature values

# Main function for prediction
if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # Load saved model weights
    model = MLPE(input_size=16 * 4 + 4)
    model.load_state_dict(torch.load('model.pth'))
    model.to(device)
    model.eval()

    # Check if preprocessed data exists
    if not os.path.exists('pre_data/pre_data.json'):
        dataset = DatasetProcess()
        dataset.save()

    # Prepare new data for prediction
    x_dataset = ChlDataset()
    data_loader = DataLoader(x_dataset, batch_size=4, shuffle=False)

    # Perform prediction
    with torch.no_grad():
        prediction = []
        for data, features in data_loader:
            output = model(data.to(device))
            output = output.detach().cpu().numpy()

            # Inverse normalization
            output[:, :300] = output[:, :300] * x_dataset.std[:300] + x_dataset.mean[:300]

            for j in range(output.shape[0]):
                prediction.append(np.concatenate((features[j].numpy(), output[j])))

        # Save predictions to CSV file
        predictions_df = pd.DataFrame(prediction)
        predictions_df.to_csv('predictions.csv', index=False, header=False)

        print('----Predict Finish----')
