import os
import torch
from PIL import Image
from torchvision import transforms


def test(model):
    path = './test/'
    imgs = []
    files = os.listdir(path)

    for name in files:
        img = Image.open(path + name).convert('L')
        img = transforms.ToTensor()(img)
        imgs.append(img)

    imgs = torch.stack(imgs, 0)

    model.eval()

    with torch.no_grad():
        output = model(imgs)

    pred = output.argmax(1)

    for i in range(len(files)):
        print(f'{files[i]}: {pred[i]}')

    print(pred)
