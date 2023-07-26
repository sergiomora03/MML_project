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
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

# Categories of the diferent lesions
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

# Save relation between label and lesion_type
label_list = df_data['label'].value_counts().keys().tolist()
lesion_list = df_data['lesion_type'].value_counts().keys().tolist()
label_to_lesion = dict(zip(label_list, lesion_list))

# Visualize some examples

lesion_groups = df_data.sort_values(["lesion_type"]).groupby("lesion_type")
num_columns = 3
num_rows = len(lesion_groups)
fig, axs = plt.subplots(num_rows, num_columns, figsize=(12, 3*num_rows))  # Adjust figsize based on the number of rows

for row, (lesion_name,images_id) in enumerate(lesion_groups):
    for col, (_, c_row) in enumerate(images_id.sample(num_columns).iterrows()):
        if col == 0:
            axs[row,col].set_ylabel(lesion_name, fontsize=20)
        ax = axs[row, col]
        ax.imshow(imread(os.path.join("data", "HAM10000_images", c_row.name + '.jpg')))
        ax.set_xticks([])  
        ax.set_yticks([])

plt.tight_layout()
fig.savefig('lesion_examples.png', dpi=300)


# Plot some histograms to caracterize the data
plt.style.use('seaborn')

plt.figure()
lesion = df_data['lesion_type'].value_counts()
plt.bar(lesion.keys().tolist(), lesion.tolist(), color=sns.color_palette("husl", 9))
plt.title('Lesion type')
plt.xticks(rotation=20)
plt.savefig('lesion_histogram.png', dpi=300)

plt.figure()
localization = df_data['localization'].value_counts()
plt.bar([x[:8] for x in localization.keys().tolist()],localization.tolist(), color=sns.color_palette("husl", 9))
plt.title('Lesion localization')
plt.xticks(rotation=30)
plt.savefig('localization_histogram.png', dpi=300)

plt.figure()
age = df_data['age'].value_counts()
plt.bar([int(x) for x in age.keys().tolist()], age.tolist(), color=sns.color_palette("husl", 9))
plt.title('Patient Age')
plt.xticks(rotation=30)
plt.savefig('age_histogram.png', dpi=300)

plt.figure()
sex = df_data['sex'].value_counts()
plt.bar(sex.keys().tolist(), sex.tolist(), color=sns.color_palette("husl", 9))
plt.title("Patient Sex")
plt.xticks(rotation=30)
plt.savefig('sex_histogram.png', dpi=300)

