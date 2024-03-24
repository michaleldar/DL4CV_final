import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset, random_split
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import wandb
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
from sklearn import metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: ", device)

VALID_LABELS = ["medical_condition"]

class CustomDataset(Dataset):
    def __init__(self, csv_file, label_to_predict="medical_condition", transform=None, image_path_column="image_path"):
        self._image_path_column = image_path_column
        self._label_to_predict = label_to_predict
        self.df = pd.read_csv(csv_file)
        self.df = self.df.drop_duplicates(subset=[self._image_path_column])
        # convert the medical condition to 0/1
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
        self.mask = transforms.ToTensor()(self.mask)
        self.mask = self.mask[0, :, :]
        self.mask = self.mask.unsqueeze(0)
        self.mask = self.mask.repeat(3, 1, 1)
        self.mask = self.mask.to(device)
        super(ApplyMaskToImage, self).__init__()

    def __call__(self, img):
        img = transforms.ToTensor()(img)
        img = img.to(device)
        img = img * self.mask
        img = transforms.ToPILImage()(img)
        return img

def custom_collate(batch):
    images, labels = zip(*batch)
    return torch.stack(images), torch.stack(labels)

def train(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    # with tqdm(total=len(train_loader), desc='Training', unit='batch') as pbar:
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

            # pbar.update(1)

    epoch_loss = running_loss / len(train_loader.dataset)
    return epoch_loss

def evaluate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    predictions = []
    y_true = []
    # pred_before_sigmoid = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            val_loss += loss.item() * inputs.size(0)
            # pred_before_sigmoid.extend(outputs.squeeze().tolist())
            predictions.extend(outputs.squeeze().tolist())
            y_true.extend(labels.tolist())
    # ROC_curve(y_true, pred_before_sigmoid)
    epoch_val_loss = val_loss / len(val_loader.dataset)
    return epoch_val_loss, predictions, y_true
#
# def ROC_curve(y_true, predictions):
#     thresholds = np.arange(0, 1, 0.01)
#     tpr = []
#     fpr = []
#     for threshold in thresholds:
#         tp = 0
#         fp = 0
#         tn = 0
#         fn = 0
#         for i in range(len(predictions)):
#             if predictions[i] >= threshold:
#                 if y_true[i] == 1:
#                     tp += 1
#                 else:
#                     fp += 1
#             else:
#                 if y_true[i] == 1:
#                     fn += 1
#                 else:
#                     tn += 1
#         tpr.append(tp / (tp + fn))
#         fpr.append(fp / (fp + tn))
#     plt.plot(fpr, tpr)
#     plt.xlabel("False Positive Rate")
#     plt.ylabel("True Positive Rate")
#     # add values of the thresholds to the plot
#     for i in range(len(thresholds)):
#         if i % 5 == 0:
#             plt.annotate(str(thresholds[i]), (fpr[i], tpr[i]))
#     plt.title("ROC Curve")
#     plt.savefig(f'/home/michalel/PycharmProjects/basic/ResNet_ft_nafld_predictions_ROC_curve.png')

# def plot_results(y_true, predictions, label):
#     y_true = np.array(y_true)
#     predictions = np.array(predictions)
#     xy = np.vstack([y_true, predictions])
#     z = gaussian_kde(xy)(xy)
#     idx = z.argsort()
#     x, y, z = y_true[idx], predictions[idx], z[idx]
#
#     fig, ax = plt.subplots()
#     ax.scatter(x, y, c=z, s=25)
#     plt.xlabel(f"Real {label}")
#     plt.ylabel(f"Predicted {label}")
#     pearson_correlation = round(pearsonr(y_true, predictions)[0], 2)
#     plt.title(f"Predicted {label} vs. Real {label}. Pearson correlation: " + str(pearson_correlation))
#
#     axes = plt.gca()
#     plt.savefig(f'/home/michalel/PycharmProjects/basic/ResNet_ft_predictions_{label}_5_epochs.png')

# def plot_correlation(correlations, label):
#     fig, ax = plt.subplots()
#     ax.plot(range(1, len(correlations) + 1), correlations)
#     plt.xlabel("Epoch")
#     plt.ylabel("Pearson Correlation")
#     plt.title("Pearson Correlation as a Function of the Epoch")
#     plt.savefig(f'/home/michalel/PycharmProjects/basic/ResNet_ft_predictions_{label}_5_epochs_correlation.png')

def main():
    label_to_predict = "medical_condition"

    wandb.init(project=label_to_predict)
    # set the run name to "3 fc hidden layers"
    wandb.run.name = "24_3_2024"


    data_transforms = {
        'train': transforms.Compose([
            ApplyMaskToImage('/home/michalel/PycharmProjects/basic/US_mask.jpg'),
            transforms.CenterCrop((720, 1000)),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]),
        'val': transforms.Compose([
            ApplyMaskToImage('/home/michalel/PycharmProjects/basic/US_mask.jpg'),
            transforms.CenterCrop((720, 1000)),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]),
    }

    csv_path = '/home/michalel/PycharmProjects/basic/us_full_dataset.csv'
    # csv_path = 'us_dataset_10_3_24.csv'
    data = CustomDataset(csv_file=csv_path, label_to_predict=label_to_predict, transform=data_transforms['train'])
    train_size = int(0.8 * len(data))
    val_size = len(data) - train_size
    # fix the seed for reproducibility
    generator1 = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(data, [train_size, val_size], generator=generator1)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    # Load pre-trained ResNet model
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features

    # Modify the model for binary classification
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 1),
        nn.Sigmoid()  # Apply sigmoid activation for binary classification
    )

    # Define loss function and optimizer for binary classification
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 15
    model.to(device)
    wandb.watch(model, log_freq=100)
    correlations = []

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        epoch_loss = train(model, train_loader, optimizer, criterion, device)
        epoch_val_loss, predictions, y_true = evaluate(model, val_loader, criterion, device)

        wandb.log({"epoch_loss": epoch_loss, "validation_loss": epoch_val_loss})
        # accuracy = accuracy_score(y_true, predictions)
        # calculate the accuracy using the sklearn roc auc score
        accuracy = metrics.roc_auc_score(y_true, predictions)
        wandb.log({"accuracy": accuracy})

        print(
            f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, Accuracy: {accuracy:.4f}')
        correlations.append(accuracy)

    # Save the trained model
    torch.save(model.state_dict(), '/home/michalel/DL4CV_final/CNN_FLD.pth')

    # Plotting results
    # plot_results(y_true, predictions, label_to_predict)
    # plot_correlation(correlations, label_to_predict)

if __name__ == '__main__':
    main()
