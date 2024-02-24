# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
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


def load_torch_model_and_evaluate(model_path):
    # load the model
    model = torch.load(model_path)
    # load the dataset
    df = create_dataset()
    # predict the values
    df['predicted_bmi'] = model.predict(df['image_path'])
    # calculate the mean absolute error
    mae = np.mean(np.abs(df['predicted_bmi'] - df['bmi']))
    # calculate the mean absolute percentage error
    mape = np.mean(np.abs(df['predicted_bmi'] - df['bmi']) / df['bmi'])
    return mae, mape

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # a = BodyMeasuresLoader.BodyMeasuresLoader().get_data(study_ids=[10])
    # df = create_dataset()
    # # plot the values of column "liver_attenuation" histogram
    # df.hist(column='liver_attenuation')
    # # plot the values of column "liver_sound_speed" as a function of "bmi" column
    # df.plot.scatter(x='bmi', y='liver_sound_speed')
    full_ds = pd.read_csv("/home/michalel/PycharmProjects/basic/us_full_dataset.csv")
    ds = full_ds[['longevity_logit', "age"]]
    # make a linear regression to predict longevity_logit from age
    model = LinearRegression()
    model.fit(ds[['age']], ds[['longevity_logit']])
    # plot the regression line
    """plt.scatter(ds[['age']], ds[['longevity_logit']])
    plt.plot(ds[['age']], model.predict(ds[['age']]), color='red')
    plt.show()"""

    """ages = [[i] for i in range(20, 90)]
    # plot from each age to the predicted longevity_logit
    plt.scatter(ages, model.predict(ages))
    plt.show()"""
    full_ds.dropna(subset=['longevity_logit'], inplace=True, how='any')
    # plot predicted longevity_logit vs. real longevity_logit
    plt.scatter(full_ds['longevity_logit'], model.predict(full_ds[['age']]))
    plt.xlabel("real longevity_logit")
    plt.ylabel("predicted longevity_logit")
    plt.show()
    # calculate pearson correlation and print
    from scipy.stats import pearsonr
    print(pearsonr(full_ds['longevity_logit'], model.predict(full_ds[['age']]).reshape(-1)))


    full_ds["longevity_logit_age_regressed_out"] = full_ds.apply(lambda x: x['longevity_logit'] - model.predict([[x['age']]])[0][0], axis=1)
    # full_ds.to_csv("/home/michalel/PycharmProjects/basic/us_full_dataset_.csv", index=False)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
