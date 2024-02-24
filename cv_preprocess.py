import cv2
import numpy as np
import os
import glob
import sys
sys.path.extend(['/home/michalel/PycharmProjects/LabData'])
sys.path.extend(['/home/michalel/PycharmProjects/LabQueue'])
sys.path.extend(['/home/michalel/PycharmProjects/LabUtils'])
from LabData.DataLoaders import SubjectLoader
from LabData.DataLoaders import BodyMeasuresLoader
import pandas as pd
from datetime import datetime


def extract_patch(image, top_left, patch_size):
    x, y = top_left
    h, w = patch_size
    return image[y:y+h, x:x+w]

def find_images_with_patch(directory, reference_patch, patch_top_left, threshold=0.9):
    # Convert the reference patch to grayscale
    reference_patch_gray = cv2.cvtColor(reference_patch, cv2.COLOR_BGR2GRAY)

    matching_images = []

    for filename in os.listdir(directory):
        if filename.lower().endswith('.jpg'):
            # Read the image
            image_path = os.path.join(directory, filename)
            image = cv2.imread(image_path)

            # Extract the patch from the same location as in the reference image
            patch = extract_patch(image, patch_top_left, reference_patch.shape[:2])

            # Convert the patch to grayscale
            patch_gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)

            # Match the reference patch with the extracted patch
            result = cv2.matchTemplate(patch_gray, reference_patch_gray, cv2.TM_CCOEFF_NORMED)

            # Check if the match is above the threshold
            if np.max(result) >= threshold:
                matching_images.append(image_path)

    return matching_images


def filter_us(search_directory='/net/mraid08/export/genie/LabData/Data/10K/aws_lab_files/ultrasound/jpg/1001201093/00_00_visit/20210826/'):
    patch_size, patch_top_left, reference_image = us_patch_id()
    # Extract the reference patch from the reference image
    reference_patch = extract_patch(reference_image, patch_top_left, patch_size)
    # Find images with similar patches in the same location
    matching_images = find_images_with_patch(search_directory, reference_patch, patch_top_left)
    return matching_images


def us_patch_id():
    reference_image_path = '/net/mraid08/export/genie/LabData/Data/10K/aws_lab_files/ultrasound/jpg/1001201093/00_00_visit/20210826/103932.jpg'
    # Define the top-left corner and size of the patch to extract from the reference image
    patch_top_left = (1130, 925)
    patch_size = (300, 155)
    reference_image = cv2.imread(reference_image_path)
    return patch_size, patch_top_left, reference_image


def _convert_to_date(date):
    if "_" in date:
        date = date.replace("_", "")
    return datetime.strptime(date, '%Y%m%d').date()


if __name__ == "__main__":
    dataset = {}
    counter = 0
    # build a dictionary of directories (/net/mraid08/export/genie/LabData/Data/10K/aws_lab_files/ultrasound/jpg/*/00_00_visit/*/*.jpg to list of images). take only 100 directories
    # for each directory, find the matching images
    for directory in glob.glob('/net/mraid08/export/genie/LabData/Data/10K/aws_lab_files/ultrasound/jpg/*/00_00_visit/*'):
        # counter += 1
        # print(directory)
        matching_images = filter_us(directory)
        # print(matching_images)
        # the key is the name of the directory under "jpg" folder
        key = "10K_" + directory.split('jpg/')[-1].split('/')[0]
        date = _convert_to_date(directory.split('jpg/')[-1].split('/')[2])
        dataset[(key, date)] = matching_images
    print(dataset)
    # save the dictionary to a file
    np.save('us_dataset_by_id_and_date.npy', dataset)
    # load the dictionary from a file:
    # dataset = np.load('us_dataset_by_id.npy', allow_pickle='TRUE').item()
    subject_data = SubjectLoader.SubjectLoader().get_data(study_ids=['10K']).df.reset_index(level=[1])
    measures_data = BodyMeasuresLoader.BodyMeasuresLoader().get_data(study_ids=['10K']).df.reset_index(level=[1])
    # change the dates in the "Date" column to contain only the date without the time
    subject_data['Date'] = subject_data['Date'].dt.date
    measures_data['Date'] = measures_data['Date'].dt.date

    # convert dataset to pandas dataframe, where the keys are the indexes and the values in a different column
    df = pd.DataFrame.from_dict(dataset, orient='index')

    # set the index tuple of df as a named list indexes
    df.index = pd.MultiIndex.from_tuples(df.index, names=['RegistrationCode', 'Date'])
    # reset the Date index to be a column, and not a MultiIndex
    df = df.reset_index(level=[1])
    # rename the column "index" to "image_path"

    # set the Date in the MultiIndex to be only date without time, and make them unique
    subject_data.index = subject_data.index.set_levels(subject_data.index.levels[1].date, level=1)

    df = df.merge(subject_data, left_index=True, right_index=True)
    df = df.merge(measures_data, left_index=True, right_index=True)
    # save the dataframe to a file
    df.to_csv('us_dataset.csv')



df['Date'] = pd.to_datetime(df.index.get_level_values('Date'))
subject_data['Date'] = pd.to_datetime(subject_data.index.get_level_values('Date'))
measures_data['Date'] = pd.to_datetime(measures_data.index.get_level_values('Date'))


# Merge dataframes on ID and Date
merged_df = pd.merge(df1, df2, left_on=['ID', 'Date'], right_on=['ID', 'Date'])


