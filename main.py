import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv_preprocess as cvp
import dataset_preprocess as dp
import CNN_FLD
import dinov2_ft_FLD as TRANSF_FLD
import visualizations


if __name__ == '__main__':
    image_path = "/net/mraid20/export/genie/LabData/Data/10K/aws_lab_files/ultrasound/jpg/1002254441/00_00_visit/20220915/092714.jpg"
    model_path = "/home/michalel/DL4CV_final/CNN_FLD.pth"
    visualizations.grad_cam(image_path, model_path)
    # dataset_path = 'us_dataset_10_3_24.csv'
    # print("Creating dataset...")
    # dataset = dp.Dataset().create_dataset(run_map_images=False)
    # print ("Dataset created, saving to csv...")
    # dataset.to_csv(dataset_path)
    # print("Predict NAFLD with CNN...")
    # CNN_FLD.main(dataset_path)
    # print("Predict NAFLD with DINOv2...")
    # TRANSF_FLD.main(dataset_path)
    # print("Done!")
