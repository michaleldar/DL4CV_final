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
from LabData.DataLoaders import BodyMeasuresLoader, SubjectLoader, UltrasoundLoader

class Dataset:
    def _init_(self):
        pass

    def extract_patch(self, image, top_left, patch_size):
        x, y = top_left
        h, w = patch_size
        return image[y:y+h, x:x+w]

    def find_images_with_patch(self, directory, reference_patch, patch_top_left, threshold=0.9):
        reference_patch_gray = cv2.cvtColor(reference_patch, cv2.COLOR_BGR2GRAY)
        matching_images = []

        for filename in os.listdir(directory):
            if filename.lower().endswith('.jpg'):
                image_path = os.path.join(directory, filename)
                image = cv2.imread(image_path)
                patch = self.extract_patch(image, patch_top_left, reference_patch.shape[:2])
                patch_gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
                result = cv2.matchTemplate(patch_gray, reference_patch_gray, cv2.TM_CCOEFF_NORMED)
                if np.max(result) >= threshold:
                    matching_images.append(image_path)
        return matching_images

    def filter_us(self, search_directory='/net/mraid20/export/genie/LabData/Data/10K/aws_lab_files/ultrasound/jpg/1001201093/00_00_visit/20210826/'):
        patch_size, patch_top_left, reference_image = self.us_patch_id()
        reference_patch = self.extract_patch(reference_image, patch_top_left, patch_size)
        matching_images = self.find_images_with_patch(search_directory, reference_patch, patch_top_left)
        return matching_images

    def us_patch_id(self):
        reference_image_path = '/net/mraid20/export/genie/LabData/Data/10K/aws_lab_files/ultrasound/jpg/1001201093/00_00_visit/20210826/103932.jpg'
        patch_top_left = (1130, 925)
        patch_size = (300, 155)
        reference_image = cv2.imread(reference_image_path)
        return patch_size, patch_top_left, reference_image

    @staticmethod
    def _convert_to_date(date):
        if "_" in date:
            date = date.replace("_", "")
        return datetime.strptime(date, '%Y%m%d').date()

    @staticmethod
    def keep_first_date(df):
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values(by='Date', ascending=True)
        df = df[~df.index.duplicated(keep='first')]
        return df

    def create_dataset(self, run_map_images=False):
        print('Creating dataset...')
        subject_data = SubjectLoader.SubjectLoader().get_data(study_ids=['10K']).df.reset_index(level=[1])
        measures_data = BodyMeasuresLoader.BodyMeasuresLoader().get_data(study_ids=['10K']).df.reset_index(level=[1])
        ultrasounds_data = UltrasoundLoader.UltrasoundLoader().get_data(study_ids=['10K']).df.reset_index(level=[1])
        subject_data['Date'] = subject_data['Date'].dt.date
        measures_data['Date'] = measures_data['Date'].dt.date
        ultrasounds_data['Date'] = ultrasounds_data['Date'].dt.date

        subject_data = self.keep_first_date(subject_data[~subject_data.index.duplicated(keep='first')])
        measures_data = self.keep_first_date(measures_data[~measures_data.index.duplicated(keep='first')])
        ultrasounds_data = self.keep_first_date(ultrasounds_data[~ultrasounds_data.index.duplicated(keep='first')])
        df = pd.merge(subject_data, measures_data, left_index=True, right_index=True)
        df = pd.merge(df, ultrasounds_data, left_index=True, right_index=True)

        # print number of rows in df
        print('Number of rows in df:', df.shape[0])
        if run_map_images:
            self.map_subject_to_images()
        images = np.load('/home/michalel/PycharmProjects/basic/map_sbj_to_img.npy', allow_pickle='TRUE').item()
        images = pd.DataFrame.from_dict(images, orient='index')
        images = images.rename(columns={0: 'image_path'})
        # print the index of images, and index of df
        print('Index of images:', images.index[:5])
        print('Index of df:', df.index[:5])
        df = df.merge(images, left_index=True, right_index=True)
        print('Number of rows in df:', df.shape[0])
        df['liver_attenuation'] = df.filter(regex='att_plus_ssp_plus_db_cm_mhz').mean(axis=1)
        df['liver_sound_speed'] = df.filter(regex='att_plus_ssp_plus_m_s').mean(axis=1)
        print('Number of rows in df:', df.shape[0])

        return df

    def map_subject_to_images(self, subjects_path_rgx='/net/mraid20/export/genie/LabData/Data/10K/aws_lab_files/ultrasound/jpg/*/00_00_visit/*', path_to_save='/home/michalel/PycharmProjects/basic/map_sbj_to_img.npy'):
        print ('Mapping subject to images...')
        dataset = {}
        count = 0
        for directory in glob.glob(subjects_path_rgx):
            count += 1
            matching_images = self.filter_us(directory)
            key = "10K_" + directory.split('jpg/')[-1].split('/')[0]
            date = self._convert_to_date(directory.split('jpg/')[-1].split('/')[2])
            dataset[(key, date)] = matching_images
            if count % 100 == 0:
                print(f'{count} subjects processed')
        np.save(path_to_save, dataset)
        print('Mapping done')


if __name__ == '__main__':
    dataset = Dataset()
    # result = dataset.create_dataset()