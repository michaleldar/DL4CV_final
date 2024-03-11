# Imports
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import accuracy_score
import wandb
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn import metrics
from datetime import datetime

# Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: ", DEVICE)
VALID_LABELS = ["medical_condition"]

# Custom classes and functions
class CustomDataset(Dataset):
    def __init__(self, csv_file, label_to_predict="medical_condition", transform=None, image_path_column="image_path"):
        self._image_path_column = image_path_column
        self._label_to_predict = label_to_predict
        self.df = pd.read_csv(csv_file).drop_duplicates(subset=[self._image_path_column])
        self.df["medical_condition"] = self.df["medical_condition"].apply(lambda x: 0 if x == "DB92" else 1)
        self.df = self.df.dropna(subset=[self._label_to_predict, self._image_path_column], how='any')
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx][self._image_path_column]
        image = Image.open(img_name).convert('RGB')
        label = torch.tensor(float(self.df.iloc[idx][self._label_to_predict]))
        if self.transform:
            image = self.transform(image)
        return image, label

class ApplyMaskToImage(object):
    def __init__(self, mask_path):
        self.mask = Image.open(mask_path).convert('RGB')
        self.mask = transforms.ToTensor()(self.mask)[0, :, :].unsqueeze(0).repeat(3, 1, 1).to(DEVICE)

    def __call__(self, img):
        img = transforms.ToTensor()(img).to(DEVICE) * self.mask
        return transforms.ToPILImage()(img)

def custom_collate(batch):
    images, labels = zip(*batch)
    return torch.stack(images), torch.stack(labels)

def train(model, train_loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(train_loader.dataset)

def evaluate(model, val_loader, criterion):
    model.eval()
    val_loss = 0.0
    predictions = []
    y_true = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            val_loss += loss.item() * inputs.size(0)
            predictions.extend(outputs.squeeze().tolist())
            y_true.extend(labels.tolist())
    return val_loss / len(val_loader.dataset), predictions, y_true

def main(dataset_path='/home/michalel/PycharmProjects/basic/us_full_dataset.csv'):
    label_to_predict = "medical_condition"
    wandb.init(project="CNN_FLD")
    # give the run a name of the date and time
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    wandb.run.name = run_name

    # Data preparation
    data_transforms = {
        'train': transforms.Compose([
            ApplyMaskToImage('/home/michalel/PycharmProjects/basic/US_mask.jpg'),
            transforms.CenterCrop((720, 1000)),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
    }

    csv_path = dataset_path
    data = CustomDataset(csv_file=csv_path, label_to_predict=label_to_predict, transform=data_transforms['train'])
    train_size = int(0.8 * len(data))
    val_size = len(data) - train_size
    train_dataset, val_dataset = random_split(data, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_dataset, batch_size=wandb.config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=wandb.config.batch_size, shuffle=False)

    # Model preparation
    model = models.resnet50(pretrained=True)
    # print(model.fc.in_features)
    # remove the last layer
    # model = nn.Sequential(*list(model.children())[:-1])
    # print the width of the last layer
    # print(model.fc.in_features)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 1),
        nn.Sigmoid()
    )
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=wandb.config.lr)

    # Training
    model.to(DEVICE)
    wandb.watch(model, log_freq=100)
    for epoch in range(wandb.config.epochs):
        train_loss = train(model, train_loader, optimizer, criterion)
        val_loss, predictions, y_true = evaluate(model, val_loader, criterion)
        accuracy = metrics.roc_auc_score(y_true, predictions)
        # calculate F1 score
        f1 = metrics.f1_score(y_true, [1 if p > 0.9 else 0 for p in predictions])
        print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}, F1 score: {f1:.4f}')

        # Log metrics
        wandb.log({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss, "accuracy": accuracy, "F1": f1})

def sweep_optimization():

    sweep_configuration = {
        "method": "bayes",
        "name": "sweep",
        "metric": {"goal": "maximize", "name": "accuracy"},
        "parameters": {
            "batch_size": {"values": [32, 64, 128]},
            "epochs": {"values": [5, 10, 15, 20]},
            "lr": {"max": 0.001, "min": 0.000001},
        },
    }
    # sweep_id = wandb.sweep(sweep_configuration, project="CNN_FLD")
    # wandb.agent(sweep_id, function=main)
    wandb.agent("60zxtel5", function=main)


if __name__ == '__main__':
    sweep_optimization()