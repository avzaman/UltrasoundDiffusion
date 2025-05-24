import pandas as pd
import os

# Load the dataset
file_path = 'train.xlsx'
df = pd.read_excel(file_path)

# Define the path to the images folder
images_folder = 'images/'

# Filter out rows where the image does not exist in the images folder
df_cleaned = df[df['Image'].apply(lambda x: os.path.isfile(os.path.join(images_folder, x)))]

# Save the cleaned dataframe to a new Excel file
df_cleaned.to_excel('train_cleaned.xlsx', index=False)

print("Rows with non-existing images have been removed and saved to 'train_cleaned.xlsx'.")
