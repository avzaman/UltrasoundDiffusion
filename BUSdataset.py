import torch
import torch.utils.data as data
from pathlib import Path
from PIL import Image
import pandas as pd
from torchvision import transforms as T

# Assuming `SimpleDataset2D` is already defined in the module
from .dataset_simple_2d import SimpleDataset2D

class BreastLesionDatasetWithLabels(SimpleDataset2D):
    def __init__(self,
                 path_root,
                 path_segmentation_labels,
                 path_classification_labels,
                 image_column='Image',
                 label_column='Label',
                 crawler_ext='png',
                 image_transform=None,
                 mask_transform=None,
                 image_resize=None,
                 augment_horizontal_flip=False,
                 augment_vertical_flip=False,
                 image_crop=None):
        """
        Custom dataset class for breast lesion images with both classification and segmentation labels.
        Args:
        - path_root: Path to the root directory containing images.
        - path_segmentation_labels: Path to the directory containing segmentation masks.
        - path_classification_labels: Path to the Excel file containing classification labels.
        - image_column: Column name in the Excel file for image names.
        - label_column: Column name in the Excel file for classification labels.
        - crawler_ext: Extension of image files to be loaded (default: 'png').
        - image_transform: Transformations to apply to the images.
        - mask_transform: Transformations to apply to the segmentation masks.
        """
        # Override the image crawler by using paths from the Excel file
        self.classification_labels = pd.read_excel(path_classification_labels, usecols=[image_column, label_column])
        self.classification_labels.set_index(image_column, inplace=True)

        # Load image pointers based on the file names in the Excel sheet
        self.item_pointers = list(self.classification_labels.index)  # This will be the list of image names without the folder path

        self.label_mapping = {'benign': 0, 'malignant': 1}
        self.path_root = Path(path_root)
        self.path_segmentation_labels = Path(path_segmentation_labels)

        # Define separate transforms for images and masks
        self.image_transform = image_transform
        self.mask_transform = mask_transform

    def __getitem__(self, index):
        # Get the image file name (without path) from the item pointers
        img_name = self.item_pointers[index]
        path_item = self.path_root / img_name

        # Load the image and segmentation mask
        img = self.load_item(path_item)
        seg_path = self.path_segmentation_labels / f"{Path(img_name).stem}.png"

        if seg_path.exists():
            seg_mask = Image.open(seg_path).convert("L")  # Load as grayscale mask
        else:
            seg_mask = Image.new("L", img.size)  # Create a blank mask if missing

        # Load classification label
        label_str = self.classification_labels.loc[img_name, 'Label']  # Get label from Excel
        classification_label = self.label_mapping[label_str]  # Convert to integer using mapping

        # Apply transformations
        if self.image_transform:
            img = self.image_transform(img)
        if self.mask_transform:
            seg_mask = self.mask_transform(seg_mask)

        return {'uid': Path(img_name).stem, 'source': img, 'segmentation': seg_mask,
                'target': classification_label}

    def load_item(self, path_item):
        """Override the load_item function to load images."""
        return Image.open(path_item).convert('RGB')



# Example usage of the custom dataset
if __name__ == "__main__":
    # Define paths
    path_to_images = "../../../../BrEaST-Lesions_USG-images_and_masks/images"
    path_to_segmentation_labels = "../../../../BrEaST-Lesions_USG-images_and_masks/labels"
    path_to_classification_labels = "../../../../BrEaST-Lesions_USG-images_and_masks/train.xlsx"

    # Define separate transformations for images and masks
    image_transform = T.Compose([
        T.Resize((224, 224)),  # Resize to 224x224
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.ToTensor(),
        T.Normalize(mean=0.5, std=0.5),  # Apply normalization only to the image
    ])

    mask_transform = T.Compose([
        T.Resize((224, 224)),  # Ensure masks have the same size as the image
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.ToTensor()  # Convert to tensor without normalization
    ])

    # Create an instance of the custom dataset with separate transforms
    breast_dataset = BreastLesionDatasetWithLabels(
        path_root=path_to_images,
        path_segmentation_labels=path_to_segmentation_labels,
        path_classification_labels=path_to_classification_labels,
        image_transform=image_transform,
        mask_transform=mask_transform,
    )

    # Check the number of samples
    print(f"Number of images in the dataset: {len(breast_dataset)}")

    # Get a sample and print its properties
    sample = breast_dataset[0]
    print(f"Sample UID: {sample['uid']}")
    print(f"Sample Image Shape: {sample['source'].shape}")
    print(f"Sample Segmentation Mask Shape: {sample['segmentation'].shape}")
    print(f"Sample Classification Label: {sample['target']}")
