import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
# import cv_preprocess as cvp
import dataset_preprocess as dp
import CNN_FLD
import dinov2_ft_FLD as TRANSF_FLD
import visualizations


if __name__ == '__main__':
    image_path = "/net/mraid20/export/genie/LabData/Data/10K/aws_lab_files/ultrasound/jpg/1002254441/00_00_visit/20220915/092714.jpg"
    FLD_images = ['/net/mraid20/export/genie/LabData/Data/10K/aws_lab_files/ultrasound/jpg/8904446287/00_00_visit/20200708/131101.jpg',
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
    non_FLD_images = ['/net/mraid20/export/genie/LabData/Data/10K/aws_lab_files/ultrasound/jpg/8373134056/00_00_visit/20200811/135758.jpg',
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

    model_path = "/home/michalel/DL4CV_final/CNN_FLD.pth"
    # visualizations.grad_cam(image_path, model_path)
    for i in range(len(FLD_images)):
        visualizations.grad_cam(FLD_images[i], model_path, True)
    for i in range(len(non_FLD_images)):
        visualizations.grad_cam(non_FLD_images[i], model_path, False)

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
