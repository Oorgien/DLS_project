import torch
import torch.nn as nn
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Preproc(nn.Module):
    def __init__(self, imsize):
        super(Preproc, self).__init__()
        self.imsize = imsize
        self.loader = transforms.Compose([
            transforms.Resize(imsize),  # нормируем размер изображения
            transforms.CenterCrop(imsize),
            transforms.ToTensor()])  # превращаем в удобный формат

        self.unloader = transforms.ToPILImage()  # тензор в кратинку

    def image_loader(self, image_name):
        image = Image.open(image_name)
        image = self.loader(image).unsqueeze(0)
        return image.to(device, torch.float)

    def transform_images(self, content, style_1, style_2):
        style1_img = self.image_loader(style_1)  # as well as here
        style2_img = self.image_loader(style_2)
        content_img = self.image_loader(content)  # измените путь на тот который у вас.
        return style1_img, style2_img, content_img

    def imshow(self, tensor, title=None):
        image = tensor.cpu().clone()
        image = image.squeeze(0)  # функция для отрисовки изображения
        image = self.unloader(image)
        plt.imshow(image)
        if title is not None:
            plt.title(title)
        plt.pause(0.001)


class Normalization(nn.Module):
        def __init__(self, mean, std):
            super(Normalization, self).__init__()
            self.mean = torch.tensor(mean).view(-1, 1, 1)
            self.std = torch.tensor(std).view(-1, 1, 1)

        def forward(self, img):
            # normalize img
            return (img - self.mean) / self.std

