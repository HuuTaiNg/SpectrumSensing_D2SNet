import torch
from prepareDataset import get_dataset
from torch.utils.data import DataLoader
from models import SDM, SSM
import torch.nn as nn
from trainingSDM import trainingAndValidation_SDM
from trainingSSM import trainingAndValidation_SSM
from torch.optim import Adam


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
print(device)
# -----

# Prepare dataset and dataloader
root = "C:\\Users\\CCE1\\Downloads\\spectrogram_tai_j03"
trainData, valData = get_dataset(root)
train_dataloader = DataLoader(trainData, batch_size=4, shuffle=True)
val_dataloader = DataLoader(valData, batch_size=16, shuffle=False)
# -----

# Prepare model
num_classes = 3
SDM = SDM()
SSM = SSM(num_classes)
def count_parameters(model):  
    return sum(p.numel() for p in model.parameters())
total_params = count_parameters(SDM) + count_parameters(SSM)
print(f"Total parameters: {total_params}")
SDM.to(device)
SSM.to(device)
SDM = nn.DataParallel(SDM)
SSM = nn.DataParallel(SSM)
# -----

# Training models
# Training SDM
criterion_SDM = nn.L1Loss()
optimizer_SDM = Adam(SDM.parameters(), lr=0.001)
numEpoch_SDM = 50
trainingAndValidation_SDM(SDM, train_dataloader, val_dataloader, criterion_SDM, optimizer_SDM, device, numEpoch_SDM)
# -----

# Training SSM
criterion_SSM = nn.CrossEntropyLoss() 
optimizer_SSM = Adam(SSM.parameters(), lr=0.001)
numEpoch_SSM = 50
SDM_path = "C:\\Users\\CCE1\\Downloads\\SDM.pt"
trainingAndValidation_SSM(SSM, SDM_path, train_dataloader, val_dataloader, criterion_SSM, optimizer_SDM, device, num_classes, numEpoch_SSM)