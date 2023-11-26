import torch
import torchvision
from CNN import CNN
from test import test
from train import train


def train_model(model, model_path, device, epochs, learning_rate, batch_size):
    print('Trenuje model... \n')

    image_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = torchvision.datasets.MNIST(
        './',
        train=True,
        download=False,
        transform=image_transform
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    loss_factor = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        train(model, device, train_loader, optimizer, epoch, loss_factor, batch_size)
        test(model)
        torch.save(model.state_dict(), model_path)


def load_existing_model(model, model_path, device):
    print('Pobieram model... \n')

    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))


def init():
    device = "cpu"
    batch_size = 128
    learning_rate = 1E-3
    model_path = "model/model.pt"
    model = CNN().to(device)

    is_from_existing_model = True if input("Czy uzyc istniejacy model? (Tak/Nie): ").lower() == "tak" else False

    if is_from_existing_model:
        load_existing_model(model, model_path, device)
    else:
        epochs = int(
            input(
                "Ilość epok (1-15, w razie podania wartosci wiekszej/mniejszej od proponowanych, bedzie brana maksymalna/minimalna): "
            )
        )

        if epochs > 15:
            epochs = 15
        elif epochs < 1:
            epochs = 1

        train_model(model, model_path, device, epochs, learning_rate, batch_size)

    test(model)


if __name__ == '__main__':
    init()
