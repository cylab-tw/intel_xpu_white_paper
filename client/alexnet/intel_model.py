import os
import torch
import warnings
from torch import nn
import torchvision.models as models
import intel_extension_for_pytorch as ipex

warnings.filterwarnings("ignore", category=UserWarning)

EPOCHS = 1
BATCH_SIZE = 64
DATA_PATH = "../dataset/PetImages"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_model():
    os.environ['TORCH_HOME'] = '.'
    alexnet_model = models.alexnet(weights='IMAGENET1K_V1')
    alexnet_model = nn.Sequential(*list(alexnet_model.children())[:-1])

    for param in alexnet_model.parameters():
        param.requires_grad = False
        
    classifier = nn.Sequential(
        nn.Flatten(),
        nn.Dropout(0.2),
        nn.Linear(256 * 6 * 6, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 2),
    )
    model = nn.Sequential(alexnet_model, classifier).to(device)

    # summary(model,(3,370, 370))
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    return model, loss_fn, optimizer

model, loss_fn, optimizer=load_model()
model, optimizer= ipex.optimize(model, optimizer=optimizer)
torch.save(model, "./model.pt")
