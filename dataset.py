import ast
import os

import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import utils as utils


class FirstImpressionsDense:
    def __init__(
            self, img_dir='./image_dir',
            df_path='./csv_dir/sparse_kfold.csv',
            stage='train', dataset='Trustworthiness', 
            numpy=True, augment=False, output_type='majority_vote', 
            test_fold=None, unique=False, fold_seed=0,
        ):
        """
        img_dir: path_to_image_dir
        df_path: path_to_"sparse_kfold.csv"
        stage: [train, validate, test] OR [0.. K-1] for KFOLD 
        dataset: dominance, trustworthiness, age, iq
        augment: True, False
        output_type: majority vote, average, distribution
        unique: only return unique faces, not unique ids, useful for test
        """
        self.dataset = dataset
        self.unique = unique
        self.stage = stage
        self.test_fold =  test_fold
        self.fold_seed = fold_seed
        self.augment = augment
        self.output_type = output_type
        self.df_path = df_path
        self.img_dir = img_dir

        self.df = self.load_csv()

    def load_csv(self):
        df = pd.read_csv(self.df_path, index_col=0, header=0)
        df = df[df['attribute'] == self.dataset]
        df = df[df['file_id'].map(lambda x: isinstance(x, str))]
        self.n_annotators = np.max(df['annotator_id'])+1

        if (self.test_fold is None) and (self.stage in ['train', 'validate', 'test']):
            df = df[df['split'] == self.stage]
        elif (self.test_fold is not None):
            train_folds = [i for i in range(5) if (i != self.test_fold)]
            istest_mask = (df['kfold_{}'.format(self.fold_seed)] == self.test_fold)
            if self.stage == 'train':
                df = df[~istest_mask]
            elif self.stage == 'test':
                df = df[istest_mask]
            else: raise ValueError
        else:
            raise ValueError

        if self.unique:
            df = df.drop_duplicates('face_id')

        return df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        i = i % len(self.df)
        row = self.df.iloc[i]
        face_id = row['face_id']
        df_face_id = self.df[self.df['face_id'] == face_id]
        annotations = df_face_id['annotation']
        imf = os.path.join(self.img_dir, row['file_id'])
        img = imageio.imread(imf)[:,:,:3]

        # cut out the image with jitter at train time
        box = int(row['x']), int(row['y']), int(row['w']), int(row['h'])
        if self.augment: box = utils.jitter_box(box, pw=0.2, ph=0.2)
        box = utils.make_box_square(box)
        roi = utils.cut_out_box(img, box, pad_mode='edge')
        roi = cv2.resize(roi, (224,224))

        if self.output_type == 'majority_vote':
            hist = np.histogram(annotations, bins=np.arange(1, 9))[0]
            annotation = np.argmax(hist)
        elif self.output_type == 'average':
            mean = np.mean(annotations)
            annotation = (mean - 1) / 6.
        elif self.output_type == 'distribution':
            hist = np.histogram(annotations, bins=np.arange(1, 9))[0]
            annotation = hist / np.sum(hist)
        else:
            raise ValueError

        ret = {
                'image': roi,
                'annotation': annotation,
                'face_id': int(row['face_id']),
        }
        return ret
