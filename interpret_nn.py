
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from captum.attr import GuidedGradCam, Deconvolution
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import to_pil_image
import shutil

import dinov2_ft_FLD

PATCH_SIZE = 8

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
    transforms.CenterCrop((650, 950)),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

preprocess2 = transforms.Compose([
        transforms.CenterCrop((650, 950)),
        transforms.Resize((224, 224)),
    ])

def visualize_grad_cam(image_path, model, is_fld=True):
    patient_id = image_path.split('/')[-4]
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    input_image = preprocess(image).unsqueeze(0)
    # plot the image after preprocessing
    plt.imshow(np.transpose(input_image.squeeze(0).detach().numpy(), (1, 2, 0)))
    plt.show()
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
    plt.imshow(np.transpose(input_image.squeeze(0).detach().numpy(), (1, 2, 0)))
    plt.title('Original Image')
    plt.subplot(1, 2, 2)
    plt.imshow(attributions, cmap='jet', alpha=0.5)
    plt.title(f'Guided Grad-CAM Heatmap for Layer4 - FLD: {is_fld}')
    # plt.show()
    plt.savefig(f'gradcam_results/grad_cam_base_result_{patient_id}_{is_fld}.jpg')

    print("predicted FLD chance: ", model(input_image))

    image = preprocess2(image)
    # convert attributions to a PIL image
    attributions = Image.fromarray(attributions)
    # take only the first channel of the attributions
    attributions = attributions.convert('L')
    # attributions = attributions.split()[0]
    # reset plot
    plt.clf()
    result = overlay_mask(image, attributions, alpha=0.5)

    result.title = "FLD positive: " + str(is_fld)
    # save the image
    result.save(f'gradcam_results/grad_cam_overlay_result_{patient_id}_{is_fld}.jpg')



def visualize_self_attention(image_path, model, is_fld=True):
    patient_id = image_path.split('/')[-4]
    for p in model.parameters():
        p.requires_grad = False

    model.eval()
    img0 = Image.open(image_path).convert('RGB')
    img = preprocess(img0)
    w, h = img.shape[1] - img.shape[1] % PATCH_SIZE, img.shape[2] - img.shape[2] % PATCH_SIZE
    img = img[:, :w, :h].unsqueeze(0)
    w_featmap = img.shape[-2] // PATCH_SIZE
    h_featmap = img.shape[-1] // PATCH_SIZE

    attentions = model.transformer.get_last_selfattention(img)  # model.get_last_selfattention(img)
    nh = attentions.shape[1]  # number of head

    # we keep only the output patch attention
    attentions = attentions[0, :, 0, 1:].reshape(nh, -1)

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
    plt.savefig(f"/home/michalel/DL4CV_final/attention_maps/attention_fld_{patient_id}_{is_fld}.png")

    image = preprocess2(img0)

    print(attentions_mean.shape)

    result = overlay_mask(image, to_pil_image(attentions_mean), alpha=0.5)

    result.title = "FLD positive: " + str(is_fld)
    # save the image
    result.save(f'attention_maps/attention_maps_overlay_result_{patient_id}_{is_fld}.jpg')


######################
def overlay_mask_2(image, attention_map, alpha=0.5):
    """Overlay attention map on the input image."""
    heatmap = plt.get_cmap('jet')(attention_map)[:, :, :3]  # Use 'jet' colormap for visualization
    heatmap = Image.fromarray((heatmap * 255).astype(np.uint8))
    heatmap = heatmap.resize(image.size, resample=Image.BILINEAR)
    result = Image.blend(image, heatmap, alpha=alpha)
    return result

def visualize_self_attention_2(image_path, model, is_fld=True):
    patient_id = image_path.split('/')[-4]
    for p in model.parameters():
        p.requires_grad = False

    model.eval()
    img0 = Image.open(image_path).convert('RGB')
    img = preprocess(img0)
    w, h = img.shape[1] - img.shape[1] % model.transformer.patch_embed.patch_size, img.shape[2] - img.shape[2] % model.transformer.patch_embed.patch_size
    img = img[:, :w, :h].unsqueeze(0)
    w_featmap = img.shape[-2] // model.transformer.patch_embed.patch_size
    h_featmap = img.shape[-1] // model.transformer.patch_embed.patch_size

    attentions = model.transformer.get_last_selfattention(img)  # Get the last layer attentions
    nh = attentions.shape[1]  # number of heads

    # we keep only the output patch attention
    attentions = attentions[0, :, 0, 1:].reshape(nh, -1)

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
    th_attn = nn.functional.interpolate(th_attn.unsqueeze(0), scale_factor=model.transformer.patch_embed.patch_size, mode="nearest")[0].cpu().numpy()

    attentions = attentions.reshape(nh, w_featmap // 2, h_featmap // 2)
    attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=model.transformer.patch_embed.patch_size, mode="nearest")[0].cpu().numpy()
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
    plt.savefig(f"/home/michalel/DL4CV_final/attention_maps/attention_fld_{patient_id}_{is_fld}.png")

    image = preprocess(img0)

    result = overlay_mask_2(image, attentions_mean, alpha=0.5)

    result.title = "FLD positive: " + str(is_fld)
    # save the image
    result.save(f'/home/michalel/DL4CV_final/attention_maps/attention_maps_overlay_result_{patient_id}_{is_fld}.jpg')


######################
dino_model = dinov2_ft_FLD.DinoVisionTransformerClassifier()
dino_model.load_state_dict(torch.load("/home/michalel/DL4CV_final/dino_model.pth"))
dino_model.eval()

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

FLD_images = [
    '/net/mraid20/export/genie/LabData/Data/10K/aws_lab_files/ultrasound/jpg/8904446287/00_00_visit/20200708/131101.jpg',
    '/net/mraid20/export/genie/LabData/Data/10K/aws_lab_files/ultrasound/jpg/9162419139/00_00_visit/20200706/133551.jpg',
    '/net/mraid20/export/genie/LabData/Data/10K/aws_lab_files/ultrasound/jpg/9282997597/00_00_visit/20210810/082608.jpg',
    '/net/mraid20/export/genie/LabData/Data/10K/aws_lab_files/ultrasound/jpg/9508626044/00_00_visit/20210314/150329.jpg',
    '/net/mraid20/export/genie/LabData/Data/10K/aws_lab_files/ultrasound/jpg/5097222583/00_00_visit/20210706/092902.jpg',
    '/net/mraid20/export/genie/LabData/Data/10K/aws_lab_files/ultrasound/jpg/4048054229/00_00_visit/20211116/115602.jpg',
    '/net/mraid20/export/genie/LabData/Data/10K/aws_lab_files/ultrasound/jpg/2224733150/00_00_visit/20220522/132503.jpg',
    '/net/mraid20/export/genie/LabData/Data/10K/aws_lab_files/ultrasound/jpg/9156196104/00_00_visit/20230323/083347.jpg',
    '/net/mraid20/export/genie/LabData/Data/10K/aws_lab_files/ultrasound/jpg/6805594980/00_00_visit/20230102/150559.jpg',
    '/net/mraid20/export/genie/LabData/Data/10K/aws_lab_files/ultrasound/jpg/5935606630/00_00_visit/20230302/103557.jpg',
    '/net/mraid20/export/genie/LabData/Data/10K/aws_lab_files/ultrasound/jpg/5028235651/00_00_visit/20230618/094549.jpg']
non_FLD_images = [
    '/net/mraid20/export/genie/LabData/Data/10K/aws_lab_files/ultrasound/jpg/8373134056/00_00_visit/20200811/135758.jpg',
    '/net/mraid20/export/genie/LabData/Data/10K/aws_lab_files/ultrasound/jpg/5686672044/00_00_visit/20210718/090718.jpg',
    '/net/mraid20/export/genie/LabData/Data/10K/aws_lab_files/ultrasound/jpg/7069017661/00_00_visit/20210812/085525.jpg',
    '/net/mraid20/export/genie/LabData/Data/10K/aws_lab_files/ultrasound/jpg/5621882640/00_00_visit/20210810/084801.jpg',
    '/net/mraid20/export/genie/LabData/Data/10K/aws_lab_files/ultrasound/jpg/7893662520/00_00_visit/20200712/105216.jpg',
    '/net/mraid20/export/genie/LabData/Data/10K/aws_lab_files/ultrasound/jpg/9363098057/00_00_visit/20200706/091820.jpg',
    '/net/mraid20/export/genie/LabData/Data/10K/aws_lab_files/ultrasound/jpg/8993073441/00_00_visit/20201111/132226.jpg',
    '/net/mraid20/export/genie/LabData/Data/10K/aws_lab_files/ultrasound/jpg/7431838179/00_00_visit/20220619/080700.jpg',
    '/net/mraid20/export/genie/LabData/Data/10K/aws_lab_files/ultrasound/jpg/4605270498/00_00_visit/20211010/104129.jpg',
    '/net/mraid20/export/genie/LabData/Data/10K/aws_lab_files/ultrasound/jpg/7182860236/00_00_visit/20230216/095354.jpg',
    '/net/mraid20/export/genie/LabData/Data/10K/aws_lab_files/ultrasound/jpg/6792297864/00_00_visit/20220323/101933.jpg']

for i in range(len(FLD_images)):
    visualize_grad_cam(FLD_images[i], model, True)
    visualize_self_attention(FLD_images[i], dino_model, True)
for i in range(len(non_FLD_images)):
    visualize_grad_cam(non_FLD_images[i], model, False)
    visualize_self_attention(non_FLD_images[i], dino_model, False)

# iterate over the images and copy them to "gradcam_results/raw" folder
# for i in range(len(FLD_images)):
#     shutil.copy(FLD_images[i], f'gradcam_results/raw/FLD_{i}.jpg')
# for i in range(len(non_FLD_images)):
#     shutil.copy(non_FLD_images[i], f'gradcam_results/raw/non_FLD_{i}.jpg')

