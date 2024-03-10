import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv_preprocess as cvp
import dataset_preprocess as dp
import CNN_FLD
import dinov2_ft_FLD as TRANSF_FLD


if __name__ == '__main__':
    dataset_path = 'us_dataset_10_3_24.csv'
    print("Creating dataset...")
    dataset = dp.Dataset().create_dataset(run_map_images=False)
    print ("Dataset created, saving to csv...")
    dataset.to_csv(dataset_path)
    print("Predict NAFLD with CNN...")
    CNN_FLD.main()
    print("Predict NAFLD with DINOv2...")
    TRANSF_FLD.main()
    print("Done!")
