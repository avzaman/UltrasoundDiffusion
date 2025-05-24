# Step 1 custom dataset and dataloader
import os
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
from sklearn.model_selection import train_test_split
import numpy as np
import torch.nn as nn
import torchvision
from torchvision import models
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def compute_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou = np.sum(intersection) / np.sum(union)
    return iou


# load dataset
def load_images_and_labels(base_path):
    images = []
    masks = []
    imagePath = os.path.join(base_path, "images")
    maskPath = os.path.join(base_path, "labels")
    for filename in os.listdir(imagePath):
        filepath = os.path.join(imagePath, filename)
        if filepath.endswith(".png"):
            images.append(filepath)
    for filename in os.listdir(maskPath):
        filepath = os.path.join(maskPath, filename)
        if filepath.endswith(".png"):
            masks.append(filepath)
    return images, masks


#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(f"Using device: {device}")

# Paths to the 'early' and 'benign' image folders
image_paths, mask_paths = load_images_and_labels(
    "./Dataset_BUSI_with_GT"
)  # Update this path

# Split dataset into train and test
train_image_paths, test_image_paths, train_mask_paths, test_mask_paths = (
    train_test_split(image_paths, mask_paths, test_size=0.2, random_state=42)
)

# Define your transformations
transformations = [
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(90),
    transforms.RandomRotation(90),
    # Add any other transformations you want to include
]

# Important CustomDataset!!
class CustomDataset(Dataset):
    def __init__(self, images, masks, transform=None, trainer = False):
        self.images = images
        self.masks = masks
        self.transform = transform
        self.trainer = trainer

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]

        # Load the original image and mask
        image = Image.open(image).convert("RGB")
        mask = Image.open(mask).resize((224, 224))

        if idx > len(self.images)/2 and self.trainer:
            # if idx is in second half of dataset do a random transofrm
            # To ensure compatibility, apply the same transformation to the mask
            # Note: For transformations like `RandomRotation`, we may need to handle it differently
            transformation = random.choice(transformations)
            if isinstance(transformation, transforms.RandomRotation):
                angle = random.choice(transformation.degrees)  # Get a random rotation angle
                image = transforms.functional.rotate(image, angle)
                mask = transforms.functional.rotate(mask, angle)
            else:
                image = transformation(image)
                mask = transformation(mask)

        mask = np.array(mask)
        mask = torch.from_numpy(mask).long()

        if self.transform:
            image = self.transform(image)

        return image, mask


# Create transforms
data_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)

# 2x the train dataset so that the second part of
train_image_paths = train_image_paths + train_image_paths
train_mask_paths = train_mask_paths + train_mask_paths

# Create an object of the dataset
train_dataset = CustomDataset(
    train_image_paths, train_mask_paths, transform=data_transform, trainer = True
)
test_dataset = CustomDataset(
    test_image_paths, test_mask_paths, transform=data_transform
)

print(len(train_dataset))
print(len(test_dataset))

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# Modify the model to use DeepLab v3
class DeepLabV3Model(nn.Module):
    def __init__(self, num_classes):
        super(DeepLabV3Model, self).__init__()

        # Load the pre-trained DeepLab v3 model from torchvision
        # Set `pretrained=True` to use the weights trained on COCO dataset
        self.deeplabv3 = torchvision.models.segmentation.deeplabv3_resnet50(
            pretrained=True
        )

        # Modify the classifier to fit the number of classes in your dataset
        # The original classifier has num_classes = 21 (for COCO)
        self.deeplabv3.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        # Pass input through the DeepLab v3 model
        x = self.deeplabv3(x)["out"]
        return x


if not os.path.exists("./breastModel2.pth"):
    num_classes = (
        2  # For example, 2 for binary segmentation (foreground and background)
    )
    model = DeepLabV3Model(num_classes=num_classes)
    model = model.to("cuda")  # Move to GPU if available
    print(next(model.parameters()).is_cuda)

    # Optimizer and Loss function
    criterion = (
        nn.CrossEntropyLoss()
    )  
    # This loss function is commonly used for segmentation
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer

    # Train model

    train_losses = []
    test_losses = []
    num_epochs = 25
    # Training loop
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_train_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to("cuda"), labels.to(
                "cuda"
            )  # Move data to the appropriate device
            
            # 1.
            optimizer.zero_grad()
            # 2. forward
            outputs = model(images)
            # 3. compute loss
            loss = criterion(outputs, labels)
            # 4. backward
            loss.backward()
            # 5. Train
            optimizer.step()  #

            running_train_loss += loss.item()

        epoch_train_loss = running_train_loss / len(train_loader)
        train_losses.append(epoch_train_loss)

        # Evaluate on test set
        model.eval()  # Set the model to evaluation mode
        running_test_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="Testing"):
                images, labels = images.to("cuda"), labels.to(
                    "cuda"
                )  # Move data to the appropriate device

                outputs = model(images)
                loss = criterion(outputs, labels)
                running_test_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.numel()
                correct += (predicted == labels).sum().item()

        epoch_test_loss = running_test_loss / len(test_loader)
        test_losses.append(epoch_test_loss)

        print(
            f"Epoch {epoch+1}, Train Loss: {epoch_train_loss:.4f}, Test Loss: {epoch_test_loss:.4f}, Test Accuracy: {100 * correct / total:.2f}%"
        )

    print("Finished Training")

    # Specify the path to save the model
    model_save_path = "breastModel2.pth"  # Change the filename if necessary

    # Save the model
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    # Plotting the loss curve
    plt.plot(range(1, num_epochs + 1), train_losses, label="Train Loss")
    plt.plot(range(1, num_epochs + 1), test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Test Loss Curve")
    plt.legend()
    save_path = os.path.join("./results-BUSI", "loss curve")
    plt.savefig(save_path)
    plt.close()
else:
    print("Skipped to eval")
    # Define your model architecture (e.g., DeepLabV3 with a ResNet backbone)
    model = DeepLabV3Model(num_classes=2)

    # Load the saved model weights
    model_path = "./modelBUSI.pth"  # Replace with your saved model path
    model.load_state_dict(torch.load(model_path))

    # Set the model to evaluation mode
    model.eval()

    # Transfer the model to the desired device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print("Model loaded successfully!")

# import cv2


# test_image = test_image_paths[0]
# Initialize counters for precision, recall, and F1 score calculations
all_true_positives = 0
all_false_positives = 0
all_false_negatives = 0
avg_iou = 0
showcount = 0
for test_image in test_image_paths:
    image = Image.open(test_image).convert("RGB")
    label = Image.open(test_image.replace("images", "labels"))

    data_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    # Convert image to tensor

    image_tensor = data_transform(image).unsqueeze(0).to("cuda")
    # print(image_tensor.shape)

    # predict using model

    predict = model(image_tensor).squeeze(0)

    predict_label = torch.argmax(predict, 0)

    # convert the tensor to numpy

    predict_label = predict_label.cpu().numpy()

    # resize the predict result to original size

    # predict_label_resize = cv2.resize(
    #    predict_label, (image.size[0], image.size[1]), interpolation=cv2.INTER_NEAREST
    # )
    # Assuming `predict_label` is your predicted label tensor

    # Convert the predicted label back to a PIL Image
    predict_label_image = Image.fromarray(predict_label.astype(np.uint8))

    # Resize the image to the original dimensions
    predict_label_resize = predict_label_image.resize(
        (image.size[0], image.size[1]), Image.NEAREST
    )

    # Convert back to numpy array if needed
    predict_label_resize = np.array(predict_label_resize)

    if showcount < 10:
        plt.subplot(1, 3, 1)
        plt.imshow(image)
        plt.subplot(1, 3, 2)
        plt.imshow(label)
        # print(predict_label_resize)
        plt.subplot(1, 3, 3)
        plt.imshow(predict_label_resize, cmap="gray")

        save_path = os.path.join("./breastimages2", f"prediction_{showcount}.jpg")
        plt.savefig(save_path)
        plt.close()
    showcount += 1
    # Compute IoU between thresh1 and label

    # IoU
    iou = compute_iou(predict_label_resize, np.array(label))
    avg_iou += iou
    print(iou)

    # Flatten both the true label and the predicted label for F1 score computation
    true_label_flat = np.array(label).flatten()
    predict_label_flat = predict_label_resize.flatten()

    # Calculate True Positives, False Positives, False Negatives
    true_positives = np.sum((predict_label_flat == 1) & (true_label_flat == 1))
    false_positives = np.sum((predict_label_flat == 1) & (true_label_flat == 0))
    false_negatives = np.sum((predict_label_flat == 0) & (true_label_flat == 1))

    # Accumulate TP, FP, FN values
    all_true_positives += true_positives
    all_false_positives += false_positives
    all_false_negatives += false_negatives

precision = all_true_positives / (
    all_true_positives + all_false_positives + 1e-7
)  # Add small epsilon to avoid division by zero
recall = all_true_positives / (all_true_positives + all_false_negatives + 1e-7)
f1 = 2 * (precision * recall) / (precision + recall + 1e-7)

print(f"Average IoU = {avg_iou/showcount}")
print(f"Precision = {precision}")
print(f"Recall = {recall}")
print(f"F1 Score = {f1}")
