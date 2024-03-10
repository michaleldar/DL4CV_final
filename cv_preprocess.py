import cv2
import numpy as np
import os
import glob
import pandas as pd
from datetime import datetime
import sys
sys.path.extend(['/home/michalel/PycharmProjects/LabData'])
sys.path.extend(['/home/michalel/PycharmProjects/LabQueue'])
sys.path.extend(['/home/michalel/PycharmProjects/LabUtils'])
from LabData.DataLoaders import BodyMeasuresLoader
from LabData.DataLoaders import SubjectLoader
from LabData.DataLoaders import UltrasoundLoader


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


def filter_us(search_directory='/net/mraid20/export/genie/LabData/Data/10K/aws_lab_files/ultrasound/jpg/1001201093/00_00_visit/20210826/'):
    patch_size, patch_top_left, reference_image = us_patch_id()
    # Extract the reference patch from the reference image
    reference_patch = extract_patch(reference_image, patch_top_left, patch_size)
    # Find images with similar patches in the same location
    matching_images = find_images_with_patch(search_directory, reference_patch, patch_top_left)
    return matching_images


def us_patch_id():
    reference_image_path = '/net/mraid20/export/genie/LabData/Data/10K/aws_lab_files/ultrasound/jpg/1001201093/00_00_visit/20210826/103932.jpg'
    # Define the top-left corner and size of the patch to extract from the reference image
    patch_top_left = (1130, 925)
    patch_size = (300, 155)
    reference_image = cv2.imread(reference_image_path)
    return patch_size, patch_top_left, reference_image


def _convert_to_date(date):
    if "_" in date:
        date = date.replace("_", "")
    return datetime.strptime(date, '%Y%m%d').date()


def keep_first_date(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by='Date', ascending=True)
    df = df[~df.index.duplicated(keep='first')]
    return df


def create_dataset(run_map_images=False):
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

    if run_map_images:
        map_subject_to_images()
    # load npy file with the dictionary of the images
    images = np.load('/home/michalel/PycharmProjects/basic/us_dataset_by_id.npy', allow_pickle='TRUE').item()
    images = pd.DataFrame.from_dict(images, orient='index')
    # set the first column to be named "image_path"
    images = images.rename(columns={0: 'image_path'})
    df = df.merge(images, left_index=True, right_index=True)
    df['liver_attenuation'] = df.filter(regex='att_plus_ssp_plus_db_cm_mhz').mean(axis=1)
    df['liver_sound_speed'] = df.filter(regex='att_plus_ssp_plus_m_s').mean(axis=1)

    return df


def map_subject_to_images(subjects_path_rgx='/net/mraid20/export/genie/LabData/Data/10K/aws_lab_files/ultrasound/jpg/*/00_00_visit/*', path_to_save='us_dataset_by_id_and_date.npy'):
    dataset = {}
    # build a dictionary of directories (/net/mraid20/export/genie/LabData/Data/10K/aws_lab_files/ultrasound/jpg/*/00_00_visit/*/*.jpg to list of images). take only 100 directories
    # for each directory, find the matching images
    for directory in glob.glob(subjects_path_rgx):
        matching_images = filter_us(directory)
        # the key is the name of the directory under "jpg" folder
        key = "10K_" + directory.split('jpg/')[-1].split('/')[0]
        date = _convert_to_date(directory.split('jpg/')[-1].split('/')[2])
        dataset[(key, date)] = matching_images
    # save the dictionary to a file
    np.save(path_to_save, dataset)


if __name__ == "__main__":
    pass
