from prepareDataset import get_dataset
import numpy as np
import torch
import matplotlib.pyplot as plt



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
root = "C:\\Users\\CCE1\\Downloads\\spectrogram_tai_j03"
trainData, valData = get_dataset(root)

image, freeNoise, label = valData[0]

image_np = image.permute(1, 2, 0).numpy()  
freeNoise_np = freeNoise.permute(1, 2, 0).numpy()  
image_np = (image_np * 255).astype(np.uint8)  

image_tensor = image.unsqueeze(0).to(device)
freeNoise_tensor = freeNoise.unsqueeze(0).to(device)
label_colored = np.zeros_like(image_np)
for rgb, idx in valData.class_colors.items():
    label_colored[label.numpy() == idx] = rgb
fig, axes = plt.subplots(1, 3, figsize=(20, 10))
axes[0].imshow(image_np)
axes[0].set_title("Image")
axes[0].axis('off')

axes[1].imshow(freeNoise_np)
axes[1].set_title("Free-noise Image")
axes[1].axis('off')

axes[2].imshow(label_colored)
axes[2].set_title("Label")
axes[2].axis('off')

plt.show()