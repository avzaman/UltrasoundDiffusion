import os
import torch
import pandas as pd
from torchvision import models
from torchvision.transforms.functional import to_pil_image
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime
from mydatasets import BreastCancerSegmentation  # Replace with your actual dataset class

def calculate_f1(pred, target):
    # Assuming pred is of shape [batch_size, 2, height, width]
    # Index 1 is the tumor class
    pred = torch.argmax(pred, dim=1)  # Get the predicted class for each pixel
    pred_tumor = (pred == 1).float()
    target_tumor = (target == 1).float()

    true_positive = (pred_tumor * target_tumor).sum(dim=(1, 2))
    false_positive = (pred_tumor * (1 - target_tumor)).sum(dim=(1, 2))
    false_negative = ((1 - pred_tumor) * target_tumor).sum(dim=(1, 2))

    precision = true_positive / (true_positive + false_positive + 1e-6)
    recall = true_positive / (true_positive + false_negative + 1e-6)

    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    f1[true_positive == 0] = 0  # Handle cases where there are no true positives

    return f1.mean()


def calculate_iou(pred, target):
    # Assuming pred is of shape [batch_size, 2, height, width]
    # Index 1 is the tumor class
    pred = torch.argmax(pred, dim=1)  # Get the predicted class for each pixel
    pred_tumor = (pred == 1).float()
    target_tumor = (target == 1).float()

    intersection = (pred_tumor * target_tumor).sum(dim=(1, 2))
    union = pred_tumor.sum(dim=(1, 2)) + target_tumor.sum(dim=(1, 2)) - intersection

    iou = intersection / (union + 1e-6)  # Add a small constant to avoid division by zero
    iou[union == 0] = 1  # Perfect match if both pred and target are all zeros

    return iou.mean()

# Helper function to save side-by-side images
def save_side_by_side_comparison(input_image, true_mask, predicted_mask, save_path):
    input_image = to_pil_image(input_image)
    true_mask = to_pil_image(true_mask)
    # predicted_mask = to_pil_image(predicted_mask)
    # Scale masks to [0, 255] and convert to PIL images
    # true_mask = to_pil_image((true_mask * 255).byte())
    # predicted_mask = to_pil_image((predicted_mask * 255).byte())
    
    # Scale true and predicted masks to [0, 1] if they are not already
    # true_mask = true_mask.float() / true_mask.max() if true_mask.max() > 1 else true_mask
    predicted_mask = predicted_mask.float() / predicted_mask.max() if predicted_mask.max() > 1 else predicted_mask

    # Convert to PIL images
    # true_mask = to_pil_image(true_mask)
    predicted_mask = to_pil_image(predicted_mask)

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(input_image)
    ax[0].set_title("Input Image")
    ax[0].axis("off")

    ax[1].imshow(true_mask, cmap="gray")
    ax[1].set_title("True Mask")
    ax[1].axis("off")

    ax[2].imshow(predicted_mask, cmap="gray")
    ax[2].set_title("Predicted Mask")
    ax[2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    matplotlib.use('Agg')
    # Set the folder and model path
    timestamp = "2024-12-07_10-55-55_USG_1"  # Modify this to match your desired folder
    base_folder = os.path.join("runs_meanteacher", timestamp)
    model_path = os.path.join(base_folder, "deeplab_meanteacher_model_0.5.pth")
    results_folder = os.path.join(base_folder, "results")
    os.makedirs(results_folder, exist_ok=True)

    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.segmentation.deeplabv3_resnet50(pretrained=True)
    model.classifier[4] = torch.nn.Conv2d(256, 2, kernel_size=(1, 1), stride=(1, 1))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # Load the test data
    base_path = "BrEaST-Lesions_USG"
    # base_path = "Dataset_BUSI_with_GT"  # Adjust base path if needed
    test_filename = "test.xlsx"  # Adjust test file name if needed
    test_data = pd.read_excel(os.path.join(base_path, test_filename))
    images_test = test_data['Image'].tolist()
    image_root = os.path.join(base_path, "images")
    test_dataset = BreastCancerSegmentation(image_root, images_test, 224, 224)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4, drop_last=True)



    # Evaluate the model\
    with torch.no_grad():
        iou_scores = []
        f1_scores = []
        precision_scores = []
        recall_scores = []

        for idx, (images, true_masks) in enumerate(test_loader):
            images, true_masks = images.to(device), true_masks.to(device)
            outputs = model(images)["out"]
            
            # Convert predicted masks to binary (0 or 1)
            predicted_masks = torch.argmax(torch.sigmoid(outputs), dim=1, keepdim=True)  # Shape: [batch_size, 1, H, W]
            # predicted_masks = (torch.sigmoid(outputs) > 0.5).float()  # Binary
            predicted_masks = predicted_masks.float()  # Convert to float for compatibility
            
            # Calculate IoU using the provided function
            iou = calculate_iou(outputs, true_masks)  # Use the provided IoU function
            iou_scores.append(iou.item())

            # Calculate F1 using the provided function
            f1 = calculate_f1(outputs, true_masks)  # Use the provided F1 function
            f1_scores.append(f1.item())

            # Save side-by-side comparisons for the first 10 batches
            if idx < 10:
                for i in range(images.size(0)):
                    save_path = os.path.join(results_folder, f"comparison_{idx * 16 + i + 1}.png")
                    save_side_by_side_comparison(
                        images[i].cpu(), 
                        true_masks[i].cpu(), 
                        predicted_masks[i].cpu(), 
                        save_path
                    )

        # Save metrics to a log file
        average_iou = sum(iou_scores) / len(iou_scores)
        average_f1 = sum(f1_scores) / len(f1_scores)

        log_file = os.path.join(results_folder, "metrics_log.txt")
        with open(log_file, "w") as f:
            f.write(f"Average IoU: {average_iou:.4f}\n")
            f.write(f"Average F1: {average_f1:.4f}\n")

        print(f"Results saved in {results_folder}")