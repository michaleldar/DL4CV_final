
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from captum.attr import GuidedGradCam, Deconvolution
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

image_path = "/net/mraid20/export/genie/LabData/Data/10K/aws_lab_files/ultrasound/jpg/1002254441/00_00_visit/20220915/092714.jpg"
class ApplyMaskToImage(object):
    def __init__(self, mask_path):
        self.mask = Image.open(mask_path).convert('RGB')
        self.mask = transforms.ToTensor()(self.mask)
        self.mask = self.mask[0, :, :]
        self.mask = self.mask.unsqueeze(0)
        self.mask = self.mask.repeat(3, 1, 1)
        # self.mask = self.mask.unsqueeze(0)
        # self.mask = self.mask.repeat(32, 1, 1, 1)
        # self.mask = self.mask.to(device)
        super(ApplyMaskToImage, self).__init__()

    def __call__(self, img):
        # apply the mask to the image
        # img = img.to(device)
        img = transforms.ToTensor()(img)
        img = img * self.mask
        # transform each image in the batch to PIL image
        img = transforms.ToPILImage()(img)

        return img


# Define transformations
preprocess = transforms.Compose([
    ApplyMaskToImage('/home/michalel/PycharmProjects/basic/US_mask.jpg'),
    # transforms.CenterCrop((720, 1000)),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load and preprocess the image
image = Image.open(image_path).convert('RGB')
input_image = preprocess(image).unsqueeze(0)

# plot the image after preprocessing

import matplotlib.pyplot as plt
import numpy as np
plt.imshow(np.transpose(input_image.squeeze(0).detach().numpy(), (1, 2, 0)))
plt.show()

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

model.load_state_dict(torch.load("/home/michalel/DL4CV_final/CNN_FLD.pth"))
model.eval()

# Define the GuidedGradCam explainer
layer = model.layer4
guided_gradcam = GuidedGradCam(model, layer)

# Generate the visualization
attributions = guided_gradcam.attribute(input_image, target=0)

# Convert the tensor to a NumPy array
attributions = attributions.squeeze(0).detach().numpy()

# Upsample the attributions for better visualization
attributions = np.abs(attributions)
attributions = np.uint8(255 * attributions / np.max(attributions))
attributions = np.array(Image.fromarray(np.transpose(attributions, (1, 2, 0))).resize((224, 224)))

# Display the original image and the heatmap
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(attributions, cmap='jet', alpha=0.5)
plt.title(f'Guided Grad-CAM Heatmap for Layer4')

# plt.show()
plt.savefig("/home/michalel/DL4CV_final/GradCam_layer_4.png")

print("predicted BMI: ", model(input_image))

