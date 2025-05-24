import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
from sklearn.model_selection import train_test_split
import torch
import numpy as np
import random
from mydatasets import BreastCancerSegmentation
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from tqdm import tqdm
from sklearn.metrics import f1_score

# Function to calculate the F1 Score
def calculate_f1_score(pred, target):
    """
    Calculates the F1 score for binary segmentation.
    Args:
        pred: Predicted tensor of shape [batch_size, height, width].
        target: Ground truth tensor of shape [batch_size, height, width].
    Returns:
        Average F1 score across the batch.
    """
    pred = torch.argmax(pred, dim=1).cpu().numpy().flatten()
    target = target.cpu().numpy().flatten()

    f1 = f1_score(target, pred, average='binary')
    return f1


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


# Modified train_model function to include F1 score calculation
def train_model(model, train_loader, criterion, optimizer, scheduler, device):
    model.train()
    running_loss = 0.0
    running_iou = 0.0
    running_f1 = 0.0

    for inputs, masks in tqdm(train_loader, desc="Training Batches"):
        inputs, masks = inputs.to(device).float(), masks.to(device).long()

        optimizer.zero_grad()
        outputs = model(inputs)['out']

        # Compute the loss
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        # Calculate IoU and F1 score for the batch
        batch_iou = calculate_iou(outputs, masks).item()
        batch_f1 = calculate_f1_score(outputs, masks)

        # Accumulate the running loss, IoU, and F1 score
        running_loss += loss.item() * inputs.size(0)
        running_iou += batch_iou * inputs.size(0)
        running_f1 += batch_f1 * inputs.size(0)

        print(f'Batch Loss: {loss.item():.4f}, Batch IoU: {batch_iou:.4f}, Batch F1: {batch_f1:.4f}')

    # Calculate epoch loss, IoU, and F1 score
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_iou = running_iou / len(train_loader.dataset)
    epoch_f1 = running_f1 / len(train_loader.dataset)

    scheduler.step()
    return epoch_loss, epoch_iou, epoch_f1

# Modified validate_model function to include F1 score calculation
def validate_model(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_iou = 0.0
    running_f1 = 0.0

    with torch.no_grad():
        for inputs, masks in tqdm(test_loader, desc="Validation Batches"):
            inputs, masks = inputs.to(device).float(), masks.to(device).long()

            outputs = model(inputs)['out']
            loss = criterion(outputs, masks)

            # Calculate IoU and F1 score for the batch
            batch_iou = calculate_iou(outputs, masks).item()
            batch_f1 = calculate_f1_score(outputs, masks)

            # Accumulate the running loss, IoU, and F1 score
            running_loss += loss.item() * inputs.size(0)
            running_iou += batch_iou * inputs.size(0)
            running_f1 += batch_f1 * inputs.size(0)

            print(f'Batch Loss: {loss.item():.4f}, Batch IoU: {batch_iou:.4f}, Batch F1: {batch_f1:.4f}')

    # Calculate epoch loss, IoU, and F1 score
    epoch_loss = running_loss / len(test_loader.dataset)
    epoch_iou = running_iou / len(test_loader.dataset)
    epoch_f1 = running_f1 / len(test_loader.dataset)

    return epoch_loss, epoch_iou, epoch_f1


if __name__ == "__main__":
    # set up random seed
    seed_value = 42
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    # torch.cuda.manual_seed_all(seed_value)  # if using multiple GPUs
    np.random.seed(seed_value)
    random.seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Load images and masks for segmentation
    # Function to load image paths and their corresponding mask paths
    def load_images_and_masks(directory):
        mask_paths = []
        for filename in os.listdir(directory):
            if filename.endswith(".png"):
                mask_paths.append(os.path.join(directory, filename))
        return mask_paths


    # Load benign and malignant images and masks
    mask = load_images_and_masks('./BrEaST-Lesions_USG/labels')

    # Split the dataset into training and testing sets
    masks_train, masks_test = train_test_split(mask, test_size=0.2, random_state=42)
    # Load training set and test set into Torch datasets
    train_dataset = BreastCancerSegmentation(masks_train, 256, 256, is_augment=True)
    test_dataset = BreastCancerSegmentation(masks_test, 256, 256, is_augment=False)

    # DataLoader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Print the number of samples in training set and testing set
    print('Training samples #: ', len(train_dataset))
    print('Test samples #: ', len(test_dataset))

    # Initialize model, criterion, optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.segmentation.deeplabv3_resnet50(pretrained=True)
    model.classifier[4] = nn.Conv2d(256, 2, kernel_size=(1, 1), stride=(1, 1))
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    # Directory to save checkpoints
    checkpoint_dir = "./checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Training loop
    num_epochs = 50
    for epoch in range(num_epochs):
        train_loss, train_iou, train_f1 = train_model(model, train_loader, criterion, optimizer, scheduler, device)
        test_loss, test_iou, test_f1 = validate_model(model, test_loader, criterion, device)

        print(f'Epoch {epoch + 1}/{num_epochs}')
        print(f'Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f}, Train F1: {train_f1:.4f}')
        print(f'Test Loss: {test_loss:.4f}, Test IoU: {test_iou:.4f}, Test F1: {test_f1:.4f}')

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")

        # Save the final model
    final_model_path = os.path.join(checkpoint_dir, "deeplabv3_usg.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved at {final_model_path}")



