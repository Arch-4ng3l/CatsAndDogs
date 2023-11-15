import torch
import torchvision
from torchvision import transforms
from PIL import Image
from os import listdir
import random
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
from os.path import isfile

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        normalize,
    ]
)

train_data_list: list[torch.Tensor] = []
target_list = []
train_data = []
dir = listdir("catsdogs/train/train/")
length = len(dir)
random.shuffle(dir)
for i in range(length):
    f = random.choice(dir)
    dir.remove(f)
    img = Image.open("catsdogs/train/train/" + f)
    img_tensor: torch.Tensor = transform(img)
    train_data_list.append(img_tensor)
    isCat = 1 if "cat" in f else 0
    isDog = 1 if "dog" in f else 0
    target = [isCat, isDog]
    target_list.append(target)
    if len(train_data_list) >= 64:
        train_data.append((torch.stack(train_data_list), target_list))
        train_data_list = []
        target_list = []
        if i > 15000:
            break
        print(f"Loaded {(i/15000)* 100}%")


class Network(nn.Module):
    def __init__(self) -> None:
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 9, kernel_size=5)
        self.conv3 = nn.Conv2d(9, 12, kernel_size=5)
        self.relu = nn.ReLU()
        self.lin1 = nn.Linear(9408, 1000)
        self.lin2 = nn.Linear(1000, 512)
        self.lin3 = nn.Linear(512, 2)

    def forward(self, X):
        X = self.conv1(X)
        X = F.max_pool2d(X, 2)
        X = self.relu(X)

        X = self.conv2(X)
        X = F.max_pool2d(X, 2)
        X = self.relu(X)

        X = self.conv3(X)
        X = F.max_pool2d(X, 2)
        X = self.relu(X)

        X = X.view(-1, 9408)
        X = self.lin1(X)
        X = F.relu(X)
        X = self.lin2(X)
        X = F.relu(X)
        X = self.lin3(X)
        return X


model = Network()
if isfile("model.pt"):
    model.load_state_dict(torch.load("model.pt"))

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.8)

criterion = nn.BCEWithLogitsLoss()


def train(epoche):
    model.train()

    for data, target in train_data:
        data = Variable(data)
        target = torch.Tensor(target)
        target = Variable(target)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        print(f"Epoche: {epoche} | Loss: {loss}")


def test():
    model.eval()

    dir = listdir("catsdogs/test/test/")
    f = random.choice(dir)
    img = Image.open("catsdogs/test/test/" + f)
    img_tensor = Variable(transform(img))
    with torch.no_grad():
        optimizer.zero_grad()
        out = model(img_tensor)
        pred_class = torch.argmax(out)

        if pred_class.item() == 0:
            print("Cat")
        else:
            print("Dog")
        img.show()
        input("Press Enter To Continue: ")


for i in range(1, 30):
    train(i)

torch.save(model.state_dict(), "model.pt")

while True:
    test()
