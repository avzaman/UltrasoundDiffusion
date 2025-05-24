import os

# Set the paths to the image and label directories
images_dir = 'images'  # Replace with your images folder path
labels_dir = 'labels'  # Replace with your labels folder path

# Get the list of all images in the images directory
image_files = set(os.listdir(images_dir))

# Iterate through the labels directory
for label_file in os.listdir(labels_dir):
    label_filename_without_extension = os.path.splitext(label_file)[0]
    # Check if the label file's corresponding image exists
    matching_image = f"{label_filename_without_extension}.png"  # Assuming images are .jpg, change if needed
    
    # If the image was deleted (i.e., it's not in the images folder), delete the corresponding label
    if matching_image not in image_files:
        label_file_path = os.path.join(labels_dir, label_file)
        os.remove(label_file_path)
        print(f"Deleted label: {label_file}")

print("Finished deleting corresponding labels for missing images.")
