import torch
import torchvision

from CNN import CNN
from test import test
from train import train


def init():
    epochs = 3
    batch_size = 128
    learning_rate = 1E-3
    device = "cpu"

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
    test_dataset = torchvision.datasets.MNIST(
        './',
        train=False,
        download=True,
        transform=image_transform
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    model = CNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    loss_factor = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        train(model, device, train_loader, optimizer, epoch, loss_factor, batch_size)



if __name__ == '__main__':
    init()
