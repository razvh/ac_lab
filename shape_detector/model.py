import os
import numpy as np
import cv2
import yaml
import torch
import torch.nn as nn
from torch.utils.data import Dataset, random_split
from skimage import io
from generate_images import generate
from torchvision import transforms
import matplotlib.pyplot as plt
names_to_numbers = {'Ellipse': 0, 'Rectangle': 1, 'Triangle': 2}
colors_to_numbers = {'blue': 0, 'green': 1, 'red': 2}
numbers_to_names = {0: 'Ellipse', 1: 'Rectangle', 2: 'Triangle'}
numbers_to_colors = {0: 'blue', 1: 'green', 2: 'red'}


class ShapesDataset(Dataset):
    def __init__(self, labels_file='labels.yaml', root_dir='dataset', source_transform=None):
        with open(labels_file, 'r') as file:
            try:
                data = yaml.safe_load(file)
            except yaml.YAMLError as exc:
                print(exc)
        self.data = data
        self.root_dir = root_dir
        self.source_transform = source_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        keys = list(self.data.keys())
        image_path = self.root_dir + "/" + keys[idx] + '.jpg'
        # print(image_path)
        image = io.imread(image_path)
        image = self.source_transform(np.array(image))
        label1 = names_to_numbers[self.data[keys[idx]][0]]
        label2 = colors_to_numbers[self.data[keys[idx]][1]]
        label3 = self.data[keys[idx]][2]
        return {'image': image, 'labels': {'label_name': label1, 'label_color': label2, 'label_area': label3}}


class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=1)
        self.batch_norm1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(2)  # 16x46x46
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
        self.batch_norm2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.batch_norm3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.batch_norm4 = nn.BatchNorm2d(64)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.shape_name = nn.Sequential(
            nn.Linear(64, 512),
            nn.Linear(512, 512),
            nn.Linear(512, 3)
        )
        self.color = nn.Sequential(
            nn.Linear(64, 64),
            nn.Linear(64, 3)
        )
        self.area = nn.Sequential(
            nn.Linear(64, 256),
            nn.Linear(256, 256),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = self.relu(self.batch_norm1(self.conv1(x)))  # 16x92x92
        x = self.max_pool(x)  # 16x46x46
        x = self.relu(self.batch_norm2(self.conv2(x)))  # 32x44x44
        x = self.max_pool(x)  # 32x22x22
        x = self.relu(self.batch_norm3(self.conv3(x)))  # 64x20x20
        x = self.max_pool(x)  # 64x10x10
        x = self.relu(self.batch_norm4(self.conv4(x)))  # 64x8x8
        x = self.avgpool(x)  # 64x1x1
        x = x.view(x.size(0), -1)
        out1 = self.shape_name(x)
        out2 = self.color(x)
        out3 = self.area(x)
        return out1, out2, out3.squeeze()


def train(epochs, lr=0.001):
    transform_train = transforms.Compose([
        transforms.ToTensor()
    ])
    generate()
    shapedata = ShapesDataset(source_transform=transform_train)
    # train_set, val_set = random_split(shapedata, [2706, 300])
    train_loader = torch.utils.data.DataLoader(shapedata, batch_size=32, shuffle=True)
    # val_loader = torch.utils.data.DataLoader(val_set, batch_size=32, shuffle=False)
    cross_entropy = nn.CrossEntropyLoss()
    mse = nn.MSELoss()
    model = MyNet()
    optimizer = torch.optim.SGD(model.parameters(), lr, momentum=0.9, weight_decay=0.0001)
    # min_valid_loss = np.inf
    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')
        loss_epoch = 0
        accuracy_out1 = 0
        accuracy_out2 = 0
        mse_out3 = 0
        total = 0
        model.train()
        for i, samples in enumerate(train_loader):
            images = samples['image']
            labels1 = samples['labels']['label_name']
            labels2 = samples['labels']['label_color']
            labels3 = samples['labels']['label_area'].float()
            out1, out2, out3 = model(images)
            loss1 = cross_entropy(out1, labels1)
            loss2 = cross_entropy(out2, labels2)
            loss3 = mse(out3, labels3)
            loss = 0.5*loss1 + 0.1*loss2 + 0.4*loss3
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                loss_epoch += loss.item()
                prediction1 = torch.argmax(out1, axis=1)
                prediction2 = torch.argmax(out2, axis=1)
                accuracy_out1 += torch.sum(prediction1 == labels1).item()
                accuracy_out2 += torch.sum(prediction2 == labels2).item()
                mse_out3 += loss3
                total += labels1.size(0)

        # valid_loss = 0
        # model.eval()
        # for i, samples in enumerate(val_loader):
        #     images = samples['image']
        #     labels1 = samples['labels']['label_name']
        #     labels2 = samples['labels']['label_color']
        #     labels3 = samples['labels']['label_area'].float()
        #     out1, out2, out3 = model(images)
        #     loss1 = cross_entropy(out1, labels1)
        #     loss2 = cross_entropy(out2, labels2)
        #     loss3 = mse(out3, labels3)
        #     loss = loss1 + loss2 + loss3
        #     valid_loss += loss.item()
        # if valid_loss < min_valid_loss:
        #     min_valid_loss = valid_loss
        #     torch.save(model.state_dict(), 'shapes_net.pth')
        print(f"Loss : {loss_epoch / len(train_loader) : .4f}")
        print(f"Accuracy_out1 : {accuracy_out1 / total : .4f}")
        print(f"Accuracy_out2 : {accuracy_out2 / total : .4f}")
        print(f"Mse_out3 : {mse_out3 / len(train_loader) : .4f}")
        print('----------------------')
        # print(f'Validation loss: {valid_loss/ len(val_loader): .4f}')
        print('----------------------')
    test(model)


def test(model, directory: str = 'test_set', n: int = 9):
    generate(directory, n, False)
    rows = 2
    cols = 5
    model.eval()
    fig = plt.figure(figsize=(8, 6))
    for i, img in enumerate(os.listdir(directory)):
        image = cv2.imread(f'{directory}/{img}')
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        out1, out2, out3 = model(image_tensor.float())
        out1 = torch.argmax(out1, axis=1).item()
        out2 = torch.argmax(out2, axis=1).item()
        out3 = out3.item()
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.set_xlabel(f"{numbers_to_names[out1]} \n {numbers_to_colors[out2]} \n {out3: .4f}")
        plt.imshow(image)
    plt.savefig('figure.png')
    plt.show()


if __name__ == '__main__':
    train(10)