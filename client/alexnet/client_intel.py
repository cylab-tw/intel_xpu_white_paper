import os, time, json
import torch
import warnings
import flwr as fl
import torchvision
from tqdm import tqdm
from PIL import Image
from torch import nn
from collections import OrderedDict
import torchvision.models as models
# import intel_extension_for_pytorch as ipex
from PIL import Image, UnidentifiedImageError
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Dataset

from torchsummary import summary
# Suppress all warnings
# warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning)

EPOCHS = 1
BATCH_SIZE = 64
DATA_PATH = "../dataset/PetImages"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CustomDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, idx):
        img_path = self.data_path[idx]
        img = Image.open(img_path)
        img = self.transform(img)
        label = 0 if "Dog" in img_path else 1
        return img, label

def load_model():
    os.environ['TORCH_HOME'] = '.'
    alexnet_model = models.alexnet(weights='IMAGENET1K_V1')
    alexnet_model = nn.Sequential(*list(alexnet_model.children())[:-1])
    
    # Freeze the parameters of the pre-trained VGG16 layers
    for param in alexnet_model.parameters():
        param.requires_grad = False

    
    # Define the custom classifier
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

def load_data():
    data_paths = []
    
    IGNORE_PATHS = [
        DATA_PATH + '/Dog/11702.jpg',
        DATA_PATH + '/Cat/666.jpg'
    ]
    classes = ['Dog', 'Cat']
    for i in classes:
        for j in os.listdir(os.path.join(DATA_PATH, i)):
            path = os.path.join(DATA_PATH, i,j)
            if path.endswith(".jpg") and path not in IGNORE_PATHS:
                data_paths.append(path)
    # print("Total Number of images:", len(set(data_paths)))

    transformation_steps = transforms.Compose([
        transforms.Resize((370, 370)),
        transforms.RandomHorizontalFlip(),  # Flips the image horizontally
        transforms.RandomRotation(20),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor()
    ])
    dataset = CustomDataset(data_path=data_paths, transform=transformation_steps)
    train_ratio = 0.8
    test_ratio = 1 - train_ratio
    
    num_train = int(train_ratio * len(dataset))
    num_test = len(dataset) - num_train
    
    train_set, test_set = random_split(dataset, [num_train, num_test])
    
    train_dataloader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)
    # print(f"Total Number of Images in Train: {len(train_dataloader) * BATCH_SIZE} | Test: {len(test_dataloader) * BATCH_SIZE} ")
    # print(f"Batch size: {BATCH_SIZE}")
    # print(f"Total Number of Images per batch in Train: {len(train_dataloader)} | Test: {len(test_dataloader)} ")
    return train_dataloader, test_dataloader


def train_model(model, train_dataloader, optimizer, epochs):
    # for epoch in range(epochs):
    model.train()
    train_loss = 0
    train_acc = 0
    train_losses, train_accuracies, train_time = [], [], []
    train = tqdm(train_dataloader)
    start_time = time.time()
    for cnt, (data, label) in enumerate(train, 1):
        data, label = data.to(device), label.to(device)
        outputs = model(data)
        loss = loss_fn(outputs, label)
        _, predict_label = torch.max(outputs, 1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_acc += (predict_label == label).sum().item()
        train_losses.append(loss.item())
        train_accuracies.append(float(train_acc) / (cnt * len(data)) * 100)
        train.set_description(f'train Epoch {epochs}')
        train.set_postfix({'loss': float(train_loss) / cnt, 'acc': float(train_acc) / (cnt * len(data)) * 100})
        end_time = time.time()
        iteration_speed = (end_time - start_time)
        train_time.append(iteration_speed)
        start_time = time.time()  # Reset start time
    return train_accuracies, train_losses, train_time

def test_model(model, test_dataloader):
    model.eval()
    test_acc = 0
    test_loss = 0
    test_losses, test_accuracies = [], []
    test = tqdm(test_dataloader)

    with torch.no_grad():
        for cnt, (data, label) in enumerate(test, 1):
            data, label = data.to(device), label.to(device)
            outputs = model(data)
            _, predict_label = torch.max(outputs, 1)
            loss = loss_fn(outputs, label)
            test_loss += loss.item()
            test_acc += (predict_label == label).sum().item()
            test_losses.append(test_loss)
            test_accuracies.append(test_acc)
            test.set_description(f'test Epoch')
            test.set_postfix({'acc': float(test_acc) / (cnt * len(data)) * 100})

    return test_loss, test_acc

model = torch.load("./model.pt")
model.eval()

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# model, loss_fn, optimizer=load_model()
# model, optimizer= ipex.optimize(model, optimizer=optimizer)
train_dataloader, test_dataloader = load_data()

class PyTorchClient(fl.client.NumPyClient):
    def __init__(self, model, train_dataloader, test_dataloader, optimizer, device):
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.optimizer = optimizer
        self.device = device

    def get_parameters(self, config):
        params = [val.cpu().numpy() for _, val in self.model.state_dict().items()]
        return params

    def set_parameters(self, parameters):
        print(parameters)
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        # print(state_dict)
        self.model.load_state_dict(state_dict, strict=True)
        print()

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train_accuracies, train_losses, train_time = train_model(self.model, self.train_dataloader, self.optimizer, EPOCHS)
        train_info = {
            "accuracies": train_accuracies[0:-1],
            "losses": train_losses[0:-1],
            "time": train_time[0:-1]
        }
        save_file = open("../data/federat/intel.json", "w")  
        json.dump(train_info, save_file, indent = 6)  
        save_file.close()
        return self.get_parameters(config={}), len(self.train_dataloader), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test_model(self.model, self.test_dataloader)
        return float(loss), len(self.test_dataloader), {"accuracy": accuracy}

fl.client.start_numpy_client(
    server_address="162.38.2.200:8080", 
    client=PyTorchClient(model, train_dataloader, test_dataloader, optimizer, device)
)