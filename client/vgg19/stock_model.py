import os
import torch
import warnings
from torch import nn
import torchvision.models as models
warnings.filterwarnings("ignore", category=UserWarning)

EPOCHS = 1
BATCH_SIZE = 64
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_model():
    os.environ['TORCH_HOME'] = '.'
    vgg19_model = models.vgg19(pretrained=True)
    
    for param in vgg19_model.parameters():
        param.requires_grad = False
    
    classifier = nn.Sequential(
        nn.Linear(25088, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 2),
    )
    
    vgg19_model.classifier = classifier
    model = vgg19_model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    return model, loss_fn, optimizer

model, loss_fn, optimizer = load_model()
torch.save(model, "./model.pt")