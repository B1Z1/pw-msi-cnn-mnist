import torch
import torchvision
import os
from CNN import CNN
from test import test
from train import train


def init():
    epochs = 3
    batch_size = 128
    learning_rate = 1E-3
    device = "cpu"
    file_path = "model/model.pt"

    model = CNN().to(device)

    if os.path.exists(file_path):
        print('Loading model... \n')
        model.load_state_dict(torch.load(file_path, map_location=torch.device('cpu')))
    else:
        print('Training model... \n')

        image_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])

        train_dataset = torchvision.datasets.MNIST(
            './',
            train=True,
            download=True,
            transform=image_transform
        )

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(model.parameters(), learning_rate)
        loss_factor = torch.nn.CrossEntropyLoss()

        for epoch in range(epochs):
            train(model, device, train_loader, optimizer, epoch, loss_factor, batch_size)
            torch.save(model.state_dict(), file_path)

    test(model)


if __name__ == '__main__':
    init()
