# Import libraries
import os, pickle
from pathlib import Path
import glob
from tqdm import tqdm
import pandas as pd
import numpy as np
from random import sample, shuffle, randint
from sklearn import metrics, model_selection
from math import ceil
import PIL, cv2
from skimage.io import imread, imshow
import matplotlib as plt
import matplotlib.patches as patches
import seaborn as sns
from torch import torch

# Categories of the diferent diseases
labels_dict = {
    'nv': 'Melanocytic nevi',
    'mel': 'Melanoma',
    'bkl': 'Benign keratosis ',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}

# Get data
csv_path = "data/HAM10000_metadata.csv"
df_data=pd.read_csv(csv_path).set_index('image_id')
df_data.dx=df_data.dx.astype('category',copy=True)
df_data['label']=df_data.dx.cat.codes # Create a new column with the encoded categories
df_data['lesion_type']= df_data.dx.map(labels_dict) # Create a new column with the lesion type
df_data.head()

breakpoint()

# Add path to images

image_path_dict = {str(x).split('/')[-1][:-4]: str(x) for x in list(glob.glob('data/*.jpg'))}

# use {filename: path} dict to select items from the correct folders
df_data['path'] = [Path(data_path/imageid_path_dict[fn].split('/')[3]/f'{fn}.jpg') for fn in df_data.index.values]

