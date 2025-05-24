import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = '7'
import torch
import numpy as np
import random
from mydatasets import BreastCancerSegmentation, BreastCancerImagesOnly
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import pandas as pd
import copy
import itertools
import datetime
import matplotlib
matplotlib.use('Agg')  # Use Agg backend to disable display
import matplotlib.pyplot as plt


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


def softmax_mse_loss(input_logits, target_logits):
    """
    Takes softmax on both sides and returns MSE loss
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = torch.nn.functional.softmax(input_logits, dim=1)
    target_softmax = torch.nn.functional.softmax(target_logits, dim=1)
    return torch.nn.functional.mse_loss(input_softmax, target_softmax)


def get_current_consistency_weight(epoch, total_epochs, initial_weight=0.01, final_weight=0.1):
    """
    Gradually increase the consistency weight over epochs
    """
    return initial_weight + (final_weight - initial_weight) * epoch / total_epochs


def consistency_rampup(epoch, max_epochs, max_weight=0.01):
    return max_weight * min(epoch / (max_epochs * 0.4), 1.0)


def train_model(model, ema_model, consistency_weight, consistency_criterion, alpha, train_loader_labeled, train_loader_unlabeled,
                criterion, optimizer, scheduler, device):
    model.train()
    ema_model.eval()  # Set teacher model to evaluation mode

    running_loss = 0.0
    running_iou = 0.0
    running_f1 = 0.0

    dataset_size = len(train_loader_labeled.dataset)
    labeled_data_iterator = itertools.cycle(train_loader_labeled)

    for unlabeled_batch in tqdm(train_loader_unlabeled, desc="Training Batches"):
        labeled_batch = next(labeled_data_iterator)
        inputs = labeled_batch[0].to(device)
        masks = labeled_batch[1].to(device)

        min_batch_size = min(inputs.size(0), unlabeled_batch[0].size(0))
        inputs = inputs[:min_batch_size]
        masks = masks[:min_batch_size]
        inputs_unlabel = unlabeled_batch[0][:min_batch_size].to(device)

        #print(f"Input shape: {inputs.shape}") # debugging
        #print(f"Mask shape: {masks.shape}") # debugging
        #print(f"Unlabled Input shape: {inputs_unlabel.shape}") # debugging
        # Forward pass
        student_outputs_labeled = model(inputs)['out']
        student_outputs_unlabeled = model(inputs_unlabel)['out']
        with torch.no_grad():
            teacher_outputs_unlabeled = ema_model(inputs_unlabel)['out']
            teacher_outputs_labeled = ema_model(inputs)['out']

        # Compute loss
        class_loss = criterion(student_outputs_labeled, masks)
        consistency_loss = consistency_weight * (consistency_criterion(student_outputs_unlabeled,
                                                                      teacher_outputs_unlabeled) +
                                                 consistency_criterion(student_outputs_labeled,
                                                                       teacher_outputs_labeled)
                                                 )
        loss = class_loss + consistency_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update the teacher model
        for teacher_param, student_param in zip(ema_model.parameters(), model.parameters()):
            teacher_param.data.mul_(alpha).add_(student_param.data, alpha=1 - alpha)

        # Update consistency weight
        consistency_weight = consistency_rampup(epoch, num_epochs, max_weight=0.001)

    running_f1 = 0.0
    with torch.no_grad():
        for inputs, masks in tqdm(train_loader_labeled, desc="Validation on train Batches"):
            inputs, masks = inputs.to(device), masks.to(device)

            outputs = ema_model(inputs)['out']
            loss = criterion(outputs, masks)

            batch_iou = calculate_iou(outputs, masks).item()
            batch_f1 = calculate_f1(outputs, masks).item()

            running_loss += loss.item() * inputs.size(0)
            running_iou += batch_iou * inputs.size(0)
            running_f1 += batch_f1 * inputs.size(0)

            # print(f'Batch Loss: {loss.item():.4f}, Batch IoU: {batch_iou:.4f}, Batch F1: {batch_f1:.4f}')

    epoch_loss = running_loss / len(train_loader_unlabeled.dataset)
    epoch_iou = running_iou / len(train_loader_labeled.dataset)
    epoch_f1 = running_f1 / len(train_loader_labeled.dataset)

    scheduler.step()
    return epoch_loss, epoch_iou, epoch_f1


def validate_model(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_iou = 0.0
    running_f1 = 0.0
    
    with torch.no_grad():
        for inputs, masks in tqdm(test_loader, desc="Validation Batches"):
            inputs, masks = inputs.to(device), masks.to(device)

            outputs = model(inputs)['out']
            loss = criterion(outputs, masks)

            batch_iou = calculate_iou(outputs, masks).item()
            batch_f1 = calculate_f1(outputs, masks).item()

            running_loss += loss.item() * inputs.size(0)
            running_iou += batch_iou * inputs.size(0)
            running_f1 += batch_f1 * inputs.size(0)

    epoch_loss = running_loss / len(test_loader.dataset)
    epoch_iou = running_iou / len(test_loader.dataset)
    epoch_f1 = running_f1 / len(test_loader.dataset)

    return epoch_loss, epoch_iou, epoch_f1

# Debug statement to check batch shapes for labeled and unlabeled data loaders
def check_batch_shapes(train_loader_labeled, train_loader_unlabeled):
    # Check labeled dataset batch shape
    for labeled_batch in train_loader_labeled:
        inputs, labels = labeled_batch
        print(f"Labeled batch input shape: {inputs.shape}, Labeled batch label shape: {labels.shape}")
        break  # Only check the first batch

    # Check unlabeled dataset batch shape
    for unlabeled_batch in train_loader_unlabeled:
        inputs_unlabeled = unlabeled_batch[0]
        print(f"Unlabeled batch input shape: {inputs_unlabeled.shape}")
        break  # Only check the first batch

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    import torchvision.models as models

    is_cuda_available = torch.cuda.is_available()

    device = torch.device("cuda" if is_cuda_available else "cpu")


    # set up random seed
    seed_value = 42
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # if using multiple GPUs
    np.random.seed(seed_value)
    random.seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Load images and masks for segmentation
    # Define the base path and filenames
    # base_path = "Dataset_BUSI_with_GT"
    base_path = "BrEaST-Lesions_USG"
    train_filename = "train.xlsx"
    test_filename = "test.xlsx"

    # Load the test datasets from the .xlsx files
    train_data = pd.read_excel(os.path.join(base_path, train_filename))
    test_data = pd.read_excel(os.path.join(base_path, test_filename))

    images_train_labeled = train_data['Image'].tolist()
    #print(images_train_labeled)
    # Extract the files
    unlabled_path = "unlabled_images"
    images_train_unlabeled = images_train_unlabeled = [f for f in os.listdir(unlabled_path) if f.endswith('.png')]
    
    print()
    #print(images_train_unlabeled)

    # Extract Image names from the test dataset
    images_test = test_data['Image'].tolist()

    # Load training set and test set into Torch datasets
    # image_root = "Dataset_BUSI_with_GT/images"
    image_root = "BrEaST-Lesions_USG/images"
    train_dataset_labeled = BreastCancerSegmentation(image_root,images_train_labeled, 224, 224)
    train_dataset_unlabeled = BreastCancerImagesOnly(unlabled_path,images_train_unlabeled, 224, 224)
    test_dataset = BreastCancerSegmentation(image_root,images_test, 224, 224)

    #print('unlabled set length: ',len(train_dataset_unlabeled))
    #print(train_dataset_unlabeled[0])

    # DataLoader
    train_loader_labeled = torch.utils.data.DataLoader(train_dataset_labeled, batch_size=32, shuffle=True, num_workers=4, drop_last=True)
    train_loader_unlabeled = torch.utils.data.DataLoader(train_dataset_unlabeled, batch_size=32, shuffle=True, num_workers=4, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, drop_last=True)

    # Example usage right after DataLoader creation
    check_batch_shapes(train_loader_labeled, train_loader_unlabeled)

    # Print the number of samples in training set and testing set
    print('Training labeled samples #: ', len(train_loader_labeled))
    print('Training unlabeled samples #: ', len(train_loader_unlabeled))

    print('Test samples #: ', len(test_dataset))

    # Initialize model, criterion, optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.segmentation.deeplabv3_resnet50(pretrained=True)
    model.classifier[4] = nn.Conv2d(256, 2, kernel_size=(1, 1), stride=(1, 1))
    model = model.to(device)
    # model = nn.DataParallel(model, device_ids=[0, 1])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    # Create a timestamped directory for saving checkpoints and the final model
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    checkpoint_dir = os.path.join("runs_meanteacher", timestamp)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Training loop
    train_losses = []
    test_losses = []

    num_epochs = 100
    for epoch in range(num_epochs):
        ema_model = copy.deepcopy(model)
        for param in ema_model.parameters():
            param.detach_()
        consistency_weight = 0

        # Mean squared error loss for consistency
        consistency_criterion = softmax_mse_loss
        alpha = 0.9999  # EMA decay factor for updating the teacher model

        train_loss, train_iou, train_f1 = train_model(
            model, ema_model, consistency_weight, consistency_criterion, alpha,
            train_loader_labeled, train_loader_unlabeled, criterion, optimizer, scheduler, device
        )
        test_loss, test_iou, test_f1 = validate_model(ema_model, test_loader, criterion, device)

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        print(f'Epoch {epoch + 1}/{num_epochs}')
        print(f'Train Loss: {train_loss:.4f} Train IoU: {train_iou:.4f} Train F1: {train_f1:.4f}')
        print(f'Test Loss: {test_loss:.4f} Test IoU: {test_iou:.4f} Test F1: {test_f1:.4f}')

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 25 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pth")
            torch.save(ema_model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")

        scheduler.step()

    # Save the final model
    final_model_path = os.path.join(checkpoint_dir, "deeplab_meanteacher_model_0.5.pth")
    torch.save(ema_model.state_dict(), final_model_path)
    print(f"Final model saved at {final_model_path}")

     # Plot and save the loss graph
    plt.figure()
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs + 1), test_losses, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train vs Test Loss')
    plt.legend()
    loss_graph_path = os.path.join(checkpoint_dir, "loss_graph.png")
    plt.savefig(loss_graph_path)
    print(f"Loss graph saved at {loss_graph_path}")
    plt.close()