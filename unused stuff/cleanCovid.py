import os

def delete_mismatched_files(image_folder, label_folder):
    # List all files in the image and label folders
    image_files = set(os.listdir(image_folder))
    label_files = set(os.listdir(label_folder))

    # Go through each image and check if a corresponding label exists
    for image_file in image_files:
        corresponding_label = f"mask_{image_file}"
        if corresponding_label not in label_files:
            print(f"Deleting image without matching mask: {image_file}")
            os.remove(os.path.join(image_folder, image_file))

    # Go through each label and check if a corresponding image exists
    for label_file in label_files:
        corresponding_image = label_file.replace("mask_", "")
        if corresponding_image not in image_files:
            print(f"Deleting mask without matching image: {label_file}")
            os.remove(os.path.join(label_folder, label_file))

# Define the folder paths
image_folder = './QaTa-COV19-v2/images'
label_folder = './QaTa-COV19-v2/labels'

# Run the mismatch check and deletion
delete_mismatched_files(image_folder, label_folder)
