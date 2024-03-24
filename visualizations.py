import torch
from torchvision.models import resnet50
from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask
from PIL import Image
import torchvision.transforms as transforms

def grad_cam(image_path, path_to_model):
    image = Image.open(image_path)
    # Load a model
    model = resnet50(pretrained=True)
    model.load_state_dict(torch.load(path_to_model))

    # Instantiate Grad-CAM
    cam_extractor = GradCAM(model, target_layer='layer4')

    # Preprocess your image
    preprocess = transforms.Compose([
        # ApplyMaskToImage('/home/michalel/PycharmProjects/basic/US_mask.jpg'),
        transforms.CenterCrop((720, 1000)),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = preprocess(image).unsqueeze(0)
    # img = preprocess(Image.open('path/to/your/image.jpg')).unsqueeze(0)

    # Extract the activation map
    target_category = 0  # None will result in the highest predicted category, originally was None. return to it?
    activation_map = cam_extractor(img, target_category)

    # Overlay and display
    # result = overlay_mask(Image.open('path/to/your/image.jpg'), activation_map, alpha=0.5)
    result = overlay_mask(image, activation_map, alpha=0.5)
    # result.show()
    # save the image
    result.save('grad_cam_result.jpg')









# FULL IMPLEMENTATION:

# import torch
# from torch.nn import functional as F
# from torchvision import models, transforms
# from PIL import Image
# import numpy as np
# import matplotlib.pyplot as plt
# import cv2

# # Load a pre-trained model
# model = models.resnet50(pretrained=True)
# model.load_state_dict(torch.load('/home/tomerse/PycharmProjects/DL4CV_final/cnn_model.pth'))
# model.eval()

# # Function to preprocess the image
# def preprocess_image(img_path):
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ])
#     image = Image.open(img_path)
#     image = transform(image).unsqueeze(0)  # Add batch dimension
#     return image

# # Function to register hook for the gradients and feature maps
# def register_hook(model):
#     gradients = []
#     activations = []
    
#     def forward_hook(module, input, output):
#         activations.append(output)
#         return None
    
#     def backward_hook(module, grad_input, grad_output):
#         gradients.append(grad_output[0])
#         return None

#     # Register hook to the last convolutional layer
#     layer = model.layer4[-1].conv1
#     layer.register_forward_hook(forward_hook)
#     layer.register_full_backward_hook(backward_hook)
    
#     return gradients, activations

# # Function to generate the Grad-CAM heatmap
# def generate_heatmap(processed_image, gradients, activations):
#     model_output = model(processed_image)
#     model_output[:, model_output.argmax()].backward()
    
#     gradient = gradients[0].cpu().data.numpy()[0]
#     activation = activations[0].cpu().data.numpy()[0]
    
#     weights = np.mean(gradient, axis=(1, 2))
#     heatmap = np.zeros(activation.shape[1:], dtype=np.float32)
    
#     for i, weight in enumerate(weights):
#         heatmap += weight * activation[i]
    
#     heatmap = np.maximum(heatmap, 0)
#     heatmap /= np.max(heatmap)
    
#     return heatmap

# # Preprocess the image
# img_path = 'path_to_your_image.jpg'  # Update this path
# processed_image = preprocess_image(img_path)

# # Register hook
# gradients, activations = register_hook(model)

# # Forward and backward pass to get gradients and activations
# heatmap = generate_heatmap(processed_image, gradients, activations)

# # Read the original image
# img = cv2.imread(img_path)
# img = cv2.resize(img, (224, 224))

# # Convert heatmap to RGB
# heatmap = np.uint8(255 * heatmap)
# heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

# # Superimpose the heatmap on original image
# superimposed_img = heatmap * 0.4 + img
# cv2.imshow('Grad-CAM', superimposed_img / superimposed_img.max())
# cv2.waitKey(0)
# cv2.destroyAllWindows()
