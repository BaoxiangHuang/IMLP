import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import orjson
import matplotlib.pyplot as plt

# Argument parser
parser = argparse.ArgumentParser(description='Chl Prediction')
parser.add_argument('--epochs', default=500, type=int, help='Number of training epochs')
parser.add_argument('--lr', default=0.00001, type=float, help='Learning rate')
parser.add_argument('--input_size', default=16 * 4 + 4, type=int, help='Input size of encoder')
parser.add_argument('--batch_size', default=4, type=int, help='Batch size')

# Gaussian Radial Basis Function Layer
class GaussianRBF(nn.Module):
    def __init__(self, input_dim):
        super(GaussianRBF, self).__init__()
        self.b = nn.Parameter(torch.randn(input_dim))  # Learnable bias

    def forward(self, x):
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
        self.file_list = os.listdir('data')
        self.save_path = 'data.json'
        self.data_path = 'data/'
        self.values = None
        self.train_data = None
        self.test_data = None
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
            useful_data = temp_data[data_index.reshape(-1), 2:]

            data = useful_data if data is None else np.concatenate((data, useful_data), axis=0)

        data[np.argwhere(data[:, 1] < 0), 1] += 360
        for i in range(data.shape[0]):
            data[i, 2] = int(str(int(data[i, 2]))[4:6])

        self.train_data, self.test_data = train_test_split(data, test_size=0.2, random_state=42)

    def save(self):
        data = {'value': self.values.values.tolist(), 'train': self.train_data.tolist(),
                'test': self.test_data.tolist()}
        with open(os.path.join(self.data_path, self.save_path), 'w') as f:
            f.write(orjson.dumps(data).decode())
        print('----Save Finish----')

# Custom Dataset
class ChlDataset(Dataset):
    def __init__(self, train):
        with open('data/data.json', 'r') as f:
            data = orjson.loads(f.read())
            self.data = np.array(data['train']).astype(float) if train else np.array(data['test']).astype(float)

        self.feature_index = [0, 1, 2, -2, -1]
        self.mean = np.mean(self.data[:, 3:-6], axis=0)
        self.std = np.std(self.data[:, 3:-6], axis=0)
        self.data[:, 3:-6] = (self.data[:, 3:-6] - self.mean) / self.std

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index, self.feature_index]
        y = self.data[index, 3:303]
        return torch.from_numpy(x), torch.from_numpy(y)

# Main function
if __name__ == '__main__':
    args = parser.parse_args()
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    net = MLPE(input_size=args.input_size).to(device)

    loss_func = nn.L1Loss().to(device)
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5)

    if not os.path.exists('data/data.json'):
        dataset = DatasetProcess()
        dataset.save()

    train_dataloader = DataLoader(ChlDataset(train=True), args.batch_size, shuffle=True)
    test_dataloader = DataLoader(ChlDataset(train=False), args.batch_size, shuffle=False)

    mean = ChlDataset(train=True).mean
    std = ChlDataset(train=True).std
    train_losses = []
    test_losses = []
    metrics = []

    best_test_loss = float('inf')
    best_model_path = 'model.pth'
    best_epoch = 0

    output_dir = 'result/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    results = []

    for epoch in range(args.epochs):
        epoch_loss = torch.zeros(1)
        for i, (data, label) in enumerate(train_dataloader):
            output = net(data.to(device))
            optimizer.zero_grad()
            loss = loss_func(output, label[:, :300].to(device))
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        train_loss = epoch_loss / len(train_dataloader)
        train_losses.append(train_loss.item())
        scheduler.step(epoch_loss)

        print(f'Processing: [{epoch + 1} / {args.epochs}] | Loss: {round(train_loss.item(), 6)} | Learning Rate: {optimizer.state_dict()["param_groups"][0]["lr"]}')

        if (epoch + 1) % 10 == 0:
            prediction = []
            test_loss = torch.zeros(1)
            net.eval()
            with torch.no_grad():
                for i, (data, label) in enumerate(test_dataloader):
                    output = net(data.to(device))
                    loss = loss_func(output, label[:, :300].to(device))
                    test_loss += loss.item()

                    output = output.detach().cpu().numpy()
                    label = label.cpu().numpy()

                    # Inverse normalization
                    output[:, :300] = output[:, :300] * std[:300] + mean[:300]
                    label[:, :300] = label[:, :300] * std[:300] + mean[:300]

                    for j in range(label.shape[0]):
                        prediction.append({
                            'features': data[j, :5].cpu().numpy().tolist(),
                            'label': label[j, :300].tolist(),
                            'predict': output[j, :300].tolist()
                        })

                test_loss = test_loss / len(test_dataloader)
                test_losses.append(test_loss.item())

                if test_loss < best_test_loss:
                    best_test_loss = test_loss
                    best_epoch = epoch + 1
                    torch.save(net.state_dict(), best_model_path)
                    print(f'New best model saved at epoch {best_epoch} with test loss: {best_test_loss.item():.6f}')

                # Save to CSV file
                epoch_file_name = os.path.join(output_dir, f'epoch_{epoch + 1}.csv')
                final_data = []
                for pred in prediction:
                    features = pred['features']
                    lab = pred['label']
                    pre = pred['predict']
                    final_data.append(features + lab)
                    final_data.append(features + pre)

                pd.DataFrame(final_data).to_csv(epoch_file_name, index=False, header=False)

    # Plot and save loss curves
    plt.plot(range(1, args.epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, args.epochs + 1), test_losses, label='Testing Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')
    plt.savefig('result/loss_curve.png')
    plt.close()

    print(f'Best test loss: {best_test_loss.item()} at epoch {best_epoch}')
