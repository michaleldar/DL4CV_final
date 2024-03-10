import sys
sys.path.extend(['/home/michalel/PycharmProjects/LabData'])
sys.path.extend(['/home/michalel/PycharmProjects/LabQueue'])
sys.path.extend(['/home/michalel/PycharmProjects/LabUtils'])
from LabData.DataLoaders import BodyMeasuresLoader
from LabData.DataLoaders import SubjectLoader
from LabData.DataLoaders import UltrasoundLoader
import pandas as pd
import numpy as np
import torch
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


def keep_first_date(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by='Date', ascending=True)
    df = df[~df.index.duplicated(keep='first')]
    return df


def create_dataset():
    subject_data = SubjectLoader.SubjectLoader().get_data(study_ids=['10K']).df.reset_index(level=[1])
    measures_data = BodyMeasuresLoader.BodyMeasuresLoader().get_data(study_ids=['10K']).df.reset_index(level=[1])
    ultrasounds_data = UltrasoundLoader.UltrasoundLoader().get_data(study_ids=['10K']).df.reset_index(level=[1])
    subject_data['Date'] = subject_data['Date'].dt.date  # Date without the time
    measures_data['Date'] = measures_data['Date'].dt.date  # Date without the time)
    ultrasounds_data['Date'] = ultrasounds_data['Date'].dt.date  # Date without the time

    # uniq duplicates by the index, keep the one with the row with the most recent date ("Date" column):
    subject_data = keep_first_date(subject_data[~subject_data.index.duplicated(keep='first')])
    measures_data = keep_first_date(measures_data[~measures_data.index.duplicated(keep='first')])
    ultrasounds_data = keep_first_date(ultrasounds_data[~ultrasounds_data.index.duplicated(keep='first')])
    # merge the dataframes
    df = pd.merge(subject_data, measures_data, left_index=True, right_index=True)
    df = pd.merge(df, ultrasounds_data, left_index=True, right_index=True)

    # load npy file with the dictionary of the images
    images = np.load('/home/michalel/PycharmProjects/basic/us_dataset_by_id.npy', allow_pickle='TRUE').item()
    images = pd.DataFrame.from_dict(images, orient='index')
    # set the first column to be named "image_path"
    images = images.rename(columns={0: 'image_path'})
    df = df.merge(images, left_index=True, right_index=True)
    df['liver_attenuation'] = df.filter(regex='att_plus_ssp_plus_db_cm_mhz').mean(axis=1)
    df['liver_sound_speed'] = df.filter(regex='att_plus_ssp_plus_m_s').mean(axis=1)

    return df


if __name__ == '__main__':
    pass
