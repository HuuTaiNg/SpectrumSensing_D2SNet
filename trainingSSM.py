from torchmetrics.classification import MulticlassJaccardIndex, MulticlassF1Score, MulticlassAccuracy, MulticlassPrecision
from torchmetrics import ConfusionMatrix
import torch
from tqdm import tqdm
import copy
import numpy as np


def prepare_for_fid(tensor):
    if tensor.shape[1] == 1:
        tensor = tensor.repeat(1, 3, 1, 1)
    tensor = tensor.clamp(0, 1)  
    tensor = (tensor*255).to(torch.uint8)
    return tensor

def trainingSSM_epoch(model, SDM, dataloader, criterion, optimizer, device, num_classes, mode="train"):
    SDM.eval()
    running_loss = 0.0
    accuracy_metric = MulticlassAccuracy(num_classes=num_classes).to(device)
    iou_metric = MulticlassJaccardIndex(num_classes=num_classes).to(device)
    precision_metric = MulticlassPrecision(num_classes=num_classes).to(device)
    f1_metric = MulticlassF1Score(num_classes=num_classes).to(device)
    confmat = ConfusionMatrix(task="multiclass", num_classes=num_classes).to(device)
    status = "Error in mode"
    if mode=="train":
        status = "Training"
        model.train()
    elif mode=="val":
        status = "Evaluating"
        model.eval()
    pbar = tqdm(dataloader, desc=status, unit='batch')
    for images, images_freenoise, labels in pbar:
        images = images.to(device)
        images_freenoise = images_freenoise.to(device)
        labels = labels.to(device)
        if mode=="train":
            outputs_SDM = SDM(images)    
            optimizer.zero_grad()
            outputs = model(images, outputs_SDM)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()   
        elif mode=="val":
            outputs_SDM = SDM(images)    
            outputs = model(images, outputs_SDM)
            loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)  
        preds = torch.argmax(outputs, dim=1)   
        accuracy_metric(preds, labels)
        iou_metric(preds, labels)
        precision_metric(preds, labels)
        f1_metric(preds, labels)
        confmat(preds, labels)  
        pbar.set_postfix({
            'Batch Loss SSM': f'{loss.item():.4f}',
            'Mean Accuracy': f'{accuracy_metric.compute():.4f}',
            'Mean IoU': f'{iou_metric.compute():.4f}',
            'Mean Precision': f'{precision_metric.compute():.4f}',
            'Mean F1 Score': f'{f1_metric.compute():.4f}'
        }) 
    epoch_loss = running_loss / len(dataloader.dataset)
    mean_accuracy = accuracy_metric.compute().cpu().numpy()
    mean_iou = iou_metric.compute().cpu().numpy()
    mean_precision = precision_metric.compute().cpu().numpy()
    mean_f1 = f1_metric.compute().cpu().numpy() 
    cm = confmat.compute().cpu().numpy() 
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]   
    return epoch_loss, mean_accuracy, mean_iou, mean_precision, mean_f1, cm_normalized

def trainingAndValidation_SSM(model, SDM_path, dataloaderTrain, dataloaderVal, criterion, optimizer, device, num_classes, num_epoch):
    epoch_saved = 0
    best_val_mAcc = 0.0  
    best_model_state = None  
    SDM = torch.jit.load(SDM_path)
    for epoch in range(num_epoch):
        epoch_loss_train, mean_accuracy_train, mean_iou_train, mean_precision_train, mean_f1_train, cm_normalized_train = trainingSSM_epoch(model, SDM, dataloaderTrain, criterion, optimizer, device, num_classes, mode="train")
        epoch_loss_val, mean_accuracy_val, mean_iou_val, mean_precision_val, mean_f1_val, cm_normalized_val = trainingSSM_epoch(model, SDM, dataloaderVal, criterion, optimizer, device, num_classes, mode="val")
        print(f"Epoch {epoch + 1}/{num_epoch}")
        print(f"Train Loss SSM: {epoch_loss_train:.4f}, Mean Accuracy: {mean_accuracy_train:.4f}, Mean IoU: {mean_iou_train:.4f}, Mean Precision: {mean_precision_train:.4f}, Mean F1: {mean_f1_train:.4f}")
        print(f"Validation Loss SSM: {epoch_loss_val:.4f}, Mean Accuracy: {mean_accuracy_val:.4f}, Mean IoU: {mean_iou_val:.4f}, Mean Precision: {mean_precision_val:.4f}, Mean F1: {mean_f1_val:.4f}")
        if mean_accuracy_val >= best_val_mAcc:
            epoch_saved = epoch + 1 
            best_val_mAcc = mean_accuracy_val
            best_model_state = copy.deepcopy(model.state_dict())       
    print("===================")
    print(f"Best SSM at epoch : {epoch_saved}")
    model.load_state_dict(best_model_state)
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    model_save = torch.jit.script(model)
    model_save.save("SSM.pt")

 