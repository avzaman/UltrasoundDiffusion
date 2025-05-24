import os
import random
import pandas as pd
from sklearn.model_selection import train_test_split

# Define paths
image_folder = "Dataset_BUSI_with_GT/images"  # Replace with your actual images folder path

# Step 1: Load all filenames with labels into a list of tuples
image_label_pairs = []

# List all image files in the folder
for filename in os.listdir(image_folder):
    if filename.endswith(('.png')):  # Adjust based on your image file types
        label = filename.split(' ')[0]  # Get the first word before the space in the filename
        if label not in ['malignant', 'benign']:
            continue  # Skip files that don't match 'malignant' or 'benign' labels
        image_label_pairs.append((filename, label))

# Step 2: Split the dataset into train and test sets
train_set, test_set = train_test_split(image_label_pairs, test_size=0.2, random_state=42)  # 80% train, 20% test

# Step 3: Create dataframes for train and test sets
train_df = pd.DataFrame(train_set, columns=['image', 'label'])
test_df = pd.DataFrame(test_set, columns=['image', 'label'])

# Step 4: Write to Excel files
train_df.to_excel('Dataset_BUSI_with_GT/train.xlsx', index=False)
test_df.to_excel('Dataset_BUSI_with_GT/test.xlsx', index=False)

print("Excel files 'train.xlsx' and 'test.xlsx' have been created successfully.")
