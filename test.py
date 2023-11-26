import os
import torch
from PIL import Image
from torchvision import transforms


def test(model):
    output_file_name = os.path.join(os.getcwd(), 'result.txt')
    test_folder_path = os.path.join(os.getcwd(), 'test')
    images = []
    files = os.listdir(test_folder_path)

    for file_name in files:
        image = Image.open(os.path.join(test_folder_path, file_name)).convert('L')
        image = transforms.ToTensor()(image)

        images.append(image)

    images = torch.stack(images, 0)

    model.eval()

    with torch.no_grad():
        output = model(images)

    pred = output.argmax(1)

    print('Results: \n')

    with open(output_file_name, 'w') as file:
        for i in range(len(files)):
            content = f'{files[i]}: {pred[i]}'

            print(content)
            file.write(content + '\n')

        file.close()
