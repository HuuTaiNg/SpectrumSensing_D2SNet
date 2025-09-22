import torch
from tqdm import tqdm
import torchmetrics
from torchmetrics.image.fid import FrechetInceptionDistance
import copy


def prepare_for_fid(tensor):
    if tensor.shape[1] == 1:
        tensor = tensor.repeat(1, 3, 1, 1)
    tensor = tensor.clamp(0, 1)  
    tensor = (tensor*255).to(torch.uint8)
    return tensor

def trainingSDM_epoch(model, dataloader, criterion, optimizer, device, mode="train"):
    running_loss = 0.0
    fid_metric = FrechetInceptionDistance().to(device)
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
            optimizer.zero_grad()
            outputs = model(images)    
            loss = criterion(255.0 * outputs, 255.0 * images_freenoise)
            loss.backward()
            optimizer.step()   
        elif mode=="val":
            outputs = model(images)    
            loss = criterion(255.0 * outputs, 255.0 * images_freenoise)

        running_loss += loss.item() * images.size(0)  
        outputs_uint8 = prepare_for_fid(outputs)
        freeNoise_uint8 = prepare_for_fid(images_freenoise)
        fid_metric.update(outputs_uint8, real=False)
        fid_metric.update(freeNoise_uint8, real=True)
        
        pbar.set_postfix({
            'Batch Loss': f'{loss.item():.4f}',
            'Mean Accuracy': f'{fid_metric.compute():.4f}',
        })
    epoch_loss = running_loss / len(dataloader.dataset)
    mean_fid = fid_metric.compute().item()
   
    return epoch_loss, mean_fid

def trainingAndValidation_SDM(model, dataloaderTrain, dataloaderVal, criterion, optimizer, device, num_epoch):
    epoch_saved = 0
    best_val_mFID = 0.0  
    best_model_state = None  
    for epoch in range(num_epoch):
        epoch_loss_train, mean_fid_train = trainingSDM_epoch(model, dataloaderTrain, criterion, optimizer, device, mode="train")
        epoch_loss_val, mean_fid_val = trainingSDM_epoch(model, dataloaderVal, criterion, optimizer, device, mode="val")
        print(f"Epoch {epoch + 1}/{num_epoch}")
        print(f"Train Loss SDM: {epoch_loss_train:.4f}, mFID: {mean_fid_train:.4f}")
        print(f"Validation Loss SDM: {epoch_loss_val:.4f}, mFID: {mean_fid_val:.4f}")
        if mean_fid_val <= best_val_mFID:
            epoch_saved = epoch + 1 
            best_val_mFID = mean_fid_val
            best_model_state = copy.deepcopy(model.state_dict())       
    print("===================")
    print(f"Best SDM at epoch : {epoch_saved}")
    model.load_state_dict(best_model_state)
    if isinstance(encoder, torch.nn.DataParallel):
        encoder = encoder.module
    model_save = torch.jit.script(encoder)
    model_save.save("SDM.pt")

 