import pandas as pd
import numpy as np
import torch
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import cv_preprocess as cvp
import dataset_preprocess as dp
import CNN_FLD


if __name__ == '__main__':
    dataset = dp.Dataset().create_dataset(run_map_images=True)
    dataset.to_csv('us_dataset_10_3_24.csv')
    CNN_FLD.main()
