import torch
from torchvision import transforms, datasets
from PIL import Image
from copy import deepcopy
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score
import wandb
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
PATCH_SIZE = 8

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


class DinoVisionTransformerClassifier(nn.Module):
    def __init__(self):
        super(DinoVisionTransformerClassifier, self).__init__()
        dinov2_vits16 = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
        self.transformer = deepcopy(dinov2_vits16)
        self.classifier = nn.Sequential(nn.Linear(384, 256), nn.ReLU(), nn.Linear(256, 1))

    def forward(self, x):
        x = self.transformer(x)
        x = self.transformer.norm(x)
        x = self.classifier(x)
        return x


class CustomDataset(Dataset):
    def __init__(self, df, transform=None, pre_transform=None, image_path_column="image_path"):
        self._image_path_column = image_path_column
        self.df = df
        self.pre_transform = pre_transform
        self.transform = transform

    def __len__(self):
        return len(self.df)

    # def __getitem__(self, idx):
    #     img_name = self.df.iloc[idx][self._image_path_column]
    #     image = Image.open(img_name).convert('RGB')
    #     label = torch.tensor(float(self.df.iloc[idx]["medical_condition"]))
    #     if self.transform:
    #         image = self.transform(image)
    #     return image, label

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx][self._image_path_column]
        image = Image.open(img_name).convert('RGB')
        label = torch.tensor(float(self.df.iloc[idx]["medical_condition"]))
        
        # Apply basic pre-transformations
        if self.pre_transform:
            image = self.pre_transform(image)
        
        # Apply EightTransforms
        images = self.transform(image) if self.transform else [image]
        
        # Convert images to tensor after applying EightTransforms
        # images = [transforms.ToTensor()(img) for img in images]

        return images, label


def visualize_self_attention(model, image_path):
    """
    Visualize the self attention of the model on the given image
    :param model:
    :param img:
    :return:
    """
    for p in model.parameters():
        p.requires_grad = False

    model.eval()
    img0 = Image.open(image_path).convert('RGB')
    img = basic_transforms(img0)
    w, h = img.shape[1] - img.shape[1] % PATCH_SIZE, img.shape[2] - img.shape[2] % PATCH_SIZE
    img = img[:, :w, :h].unsqueeze(0)
    w_featmap = img.shape[-2] // PATCH_SIZE
    h_featmap = img.shape[-1] // PATCH_SIZE

    attentions = model.transformer.get_last_selfattention(img)  # model.get_last_selfattention(img)
    nh = attentions.shape[1]  # number of head

    # we keep only the output patch attention
    attentions = attentions[0, :, 0, 1:].reshape(nh, -1)
    print(attentions.shape)

    # we keep only a certain percentage of the mass
    val, idx = torch.sort(attentions)
    val /= torch.sum(val, dim=1, keepdim=True)
    cumval = torch.cumsum(val, dim=1)
    threshold = 0.6  # We visualize masks obtained by thresholding the self-attention maps to keep xx% of the mass.
    th_attn = cumval > (1 - threshold)
    idx2 = torch.argsort(idx)
    for head in range(nh):
        th_attn[head] = th_attn[head][idx2[head]]

    th_attn = th_attn.reshape(nh, w_featmap // 2, h_featmap // 2).float()

    # interpolate
    th_attn = nn.functional.interpolate(th_attn.unsqueeze(0), scale_factor=PATCH_SIZE, mode="nearest")[0].cpu().numpy()

    attentions = attentions.reshape(nh, w_featmap // 2, h_featmap // 2)
    attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=PATCH_SIZE, mode="nearest")[
        0].cpu().numpy()
    attentions_mean = np.mean(attentions, axis=0)
    plt.figure(figsize=(6, 6), dpi=200)

    plt.subplot(3, 3, 1)
    plt.title("Original", size=6)
    plt.imshow(img0)
    plt.axis("off")

    plt.subplot(3, 3, 2)
    plt.title("Attentions Mean", size=6)
    plt.imshow(attentions_mean)
    plt.axis("off")

    for i in range(6):
        plt.subplot(3, 3, i + 4)
        plt.title("Attentions " + str(i), size=6)
        plt.imshow(attentions[i])
        plt.axis("off")
    rnd_text = str(np.random.randint(100000))
    print("random: ", rnd_text)
    plt.savefig(f"/home/michalel/PycharmProjects/basic/attention_fld_{rnd_text}.png")


class EightTransforms:
  """Generate all the possible crops using combinations of
  [90, 180, 270 degrees rotations,  horizontal flips and vertical flips]. 
  In total there are 8 options."""

  def __init__(self):
    pass

  def __call__(self, sample):
    """
    Args:
      sample (torch.Tensor) - image to be transformed.
      Has shape `(num_channels, height, width)`.
    Returns:
      output (List(torch.Tensor)) - A list of 8 tensors containing the different
      flips and rotations of the original image. Each tensor has the same size as 
      the original image, possibly transposed in the spatial dimensions.
    """

    flipped = torch.flip(sample, [2])
    output = self._get_rotations(sample) + self._get_rotations(flipped)
    return output

  def _get_rotations(self, sample):
    return [
            sample,
            torch.rot90(sample, 1, [1, 2]),
            torch.rot90(sample, 2, [1, 2]),
            torch.rot90(sample, 3, [1, 2]),
    ]



basic_transforms = transforms.Compose([
    ApplyMaskToImage('/home/michalel/PycharmProjects/basic/US_mask.jpg'),
    transforms.CenterCrop((720, 1000)),
    transforms.Resize((480, 480)),
    transforms.ToTensor()
])


# data_transforms = {
#         "train": transforms.Compose(
#             [
#                 ApplyMaskToImage('/home/michalel/PycharmProjects/basic/US_mask.jpg'),
#                 transforms.CenterCrop((720, 1000)),
#                 # transforms.Resize((224, 224)),
#                 transforms.Resize((480, 480)),
#                 transforms.RandomRotation(360),
#                 transforms.RandomHorizontalFlip(),
#                 transforms.RandomVerticalFlip(),
#                 transforms.ToTensor()
#             ]
#         ),
#         "validation": transforms.Compose(
#             [
#                 ApplyMaskToImage('/home/michalel/PycharmProjects/basic/US_mask.jpg'),
#                 transforms.CenterCrop((720, 1000)),
#                 # transforms.Resize((224, 224)),
#                 transforms.Resize((480, 480)),
#                 # transforms.RandomRotation(360),
#                 # transforms.RandomHorizontalFlip(),
#                 # transforms.RandomVerticalFlip(),
#                 transforms.ToTensor()
#             ]
#         ),
#     }


def main(dataset_path='/home/michalel/PycharmProjects/basic/us_full_dataset.csv'):
    wandb.init(project="DINOv2_FLD")
    image_path_column="image_path"
    df = pd.read_csv(dataset_path)
    df = df.drop_duplicates(subset=[image_path_column])
    # convert the medical condition to 0/1
    df["medical_condition"] = df["medical_condition"].apply(lambda x: 0 if x == "DB92" else 1)
    df = df.dropna(subset=["medical_condition", image_path_column], how='any')
    # split the data into train and validation using pandas
    train_size = int(0.8 * len(df))
    val_size = len(df) - train_size
    # fix the seed for reproducibility
    generator1 = torch.Generator().manual_seed(42)
    train_split, val_split = random_split(df, [train_size, val_size], generator=generator1)

    train_split = train_split.dataset.iloc[train_split.indices]
    val_split = val_split.dataset.iloc[val_split.indices]

    # train_data = CustomDataset(train_split, transform=data_transforms['train'])
    # val_data = CustomDataset(val_split, transform=data_transforms['validation'])
    train_data = CustomDataset(train_split, transform=EightTransforms(), pre_transform=basic_transforms)
    val_data = CustomDataset(val_split, pre_transform=basic_transforms) 
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False, num_workers=0)
    model = DinoVisionTransformerClassifier()
    model = model.to(device)
    model = model.train()
    wandb.watch(model, log_freq=100)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-6)
    num_epochs = 15
    epoch_losses = []
    for epoch in range(num_epochs):
        print("Epoch: ", epoch)

        batch_losses = []

        for data in train_loader:
            # Unpack the data
            list_of_images_batch, labels = data
            labels = labels.to(device)  # Assuming labels is a tensor of shape [batch_size]

            # Since all images in the list have the same label, we can process each
            # image in the list one by one or in smaller batches if needed.
            for images in list_of_images_batch:
                # images = torch.stack(images).to(device)  # Stack images to form a new batch
                images = images.to(device)
                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass: Compute predicted output by passing images to the model
                outputs = model(images)

                # Calculate the loss
                # Note: You might need to adjust the label tensor shape depending on your loss function's expectation
                # If your loss function expects labels to have the same batch size as outputs,
                # you might need to expand or repeat the labels tensor.
                print(labels.shape)
                print(labels.expand_as(outputs).shape)
                print(outputs.shape)
                loss = criterion(outputs, labels.expand_as(outputs).squeeze())



                # Backward pass: Compute gradient of the loss with respect to model parameters
                loss.backward()

                # Perform a single optimization step (parameter update)
                optimizer.step()

                batch_losses.append(loss.item())
        # for data in train_loader:
        #     # get the input batch and the labels
        #     batch_of_images, labels = data

        #     # zero the parameter gradients
        #     optimizer.zero_grad()

        #     # model prediction
        #     output = model(batch_of_images.to(device)).squeeze(dim=1)

        #     # compute loss and do gradient descent
        #     loss = criterion(output, labels.float().to(device))
        #     loss.backward()
        #     optimizer.step()

        #     batch_losses.append(loss.item())

        epoch_losses.append(np.mean(batch_losses))
        print(f"Mean epoch loss: {epoch_losses[-1]}")
        # evaluate the model on the validation set
        model = model.eval()
        predictions = []
        y_true = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                predictions.extend(outputs.squeeze().tolist())
                y_true.extend(labels.tolist())
            predictions = np.array(predictions)
            y_true = np.array(y_true)
            val_loss = criterion(torch.tensor(predictions).to(device), torch.tensor(y_true).float().to(device))
        print(f"Validation loss: {val_loss.item()}")
        accuracy = np.mean(np.round(predictions) == y_true)
        f1 = f1_score(y_true, [1 if p > 0.9 else 0 for p in predictions])
        roc_auc = roc_auc_score(y_true, predictions)
        print(f"Validation accuracy: {accuracy}", f"ROC AUC: {roc_auc}, F1: {f1}")
        wandb.log({"epoch_loss": epoch_losses[-1],
                   "validation_loss": val_loss.item(),
                   "accuracy": accuracy, "roc_auc": roc_auc, "F1": f1})
        return model


if __name__ == '__main__':
    model = main()

