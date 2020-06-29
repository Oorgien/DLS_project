from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

import copy
import deepmux

import requests
from IPython.display import clear_output

from Images import Preproc, Normalization

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def gram_matrix(input):
        batch_size , h, w, f_map_num = input.size()
        features = input.view(batch_size * h, w * f_map_num)
        G = torch.mm(features, features.t())
        return G.div(batch_size * h * w * f_map_num)

class ContentLoss(nn.Module):

        def __init__(self, target,):
            super(ContentLoss, self).__init__()
            self.target = target.detach()
            self.loss = F.mse_loss(self.target, self.target )

        def forward(self, input):
            self.loss = F.mse_loss(input, self.target)
            return input


class StyleLoss(nn.Module):
        def __init__(self, target_feature):
            super(StyleLoss, self).__init__()
            self.target = gram_matrix(target_feature).detach()
            self.loss = F.mse_loss(self.target, self.target)# to initialize with something

        def forward(self, input):
            G = gram_matrix(input)
            self.loss = F.mse_loss(G, self.target)
            return input

class Style_transfer(nn.Module):
    cnn = models.vgg19(pretrained=True).to(device).features.eval()
    
    content_layers_default = ['conv_4']
    style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
    
    normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
    
    def __init__(self, content_img, style1_img, style2_img, input_img, num_steps=500,
                          style_weight_1=100000,style_weight_2=100000, content_weight=1, mode = 2):
        super().__init__()
        self.content_img = content_img
        self.style1_img = style1_img
        self.style2_img = style2_img
        self.input_img = input_img
        self.num_steps = num_steps
        self.style_weight_1 = style_weight_1
        self.style_weight_2 = style_weight_2
        self.content_weight = content_weight
        self.mode = mode
        
    def get_input_optimizer(self,input_img):
        # this line to show that input is a parameter that requires a gradient
        #добоваляет содержимое тензора катринки в список изменяемых оптимизатором параметров
        optimizer = optim.LBFGS([input_img.requires_grad_()]) 
        return optimizer
    
    def forward(self):
        """Run the style transfer."""
        print('Building the style transfer model..')
        model, style_losses_1,style_losses_2, content_losses = self.get_style_model_and_losses()
        optimizer = self.get_input_optimizer(self.input_img)
        print('Optimizing..')
        run = [0]
        while run[0] <= self.num_steps:
             def closure():
                # correct the values 
                # это для того, чтобы значения тензора картинки не выходили за пределы [0;1]
                self.input_img.data.clamp_(0, 1)
                optimizer.zero_grad()
                model(self.input_img)
                self.input_img.data.clamp_(0, 1)
                style_score_1 = 0
                style_score_2 = 0
                content_score = 0
                
                for sl in style_losses_1:
                    style_score_1 += sl.loss
                if self.mode == 2:
                    for sl in style_losses_2:
                        style_score_2 += sl.loss
                for cl in content_losses:
                    content_score += cl.loss
                    
                style_score_1 *= self.style_weight_1
                if self.mode == 2:
                    style_score_2 *= self.style_weight_2
                content_score *= self.content_weight
                
                loss = style_score_1 + style_score_2 + content_score
                loss.backward()
                
                run[0] += 1
                if run[0] % 50 == 0:
                    clear_output(wait=True)
                    print("run {}:".format(run))
                    if self.mode == 2:
                        print('Style Loss 1: {:4f} Style Loss 2: {:4f} Content Loss: {:4f}'.format(
                            style_score_1.item(), style_score_2.item(),content_score.item()))
                    else:
                        print('Style Loss 1: {:4f} Content Loss: {:4f}'.format(
                            style_score_1.item(),content_score.item()))
                return style_score_1 + style_score_2 + content_score
             optimizer.step(closure)
        self.input_img.data.clamp_(0, 1)
        return self.input_img
    
    def get_style_model_and_losses(self):
        cnn = copy.deepcopy(self.cnn)

        normalization = Normalization(self.normalization_mean, self.normalization_std).to(device)

        content_losses = []
        style_losses_1 = []
        style_losses_2 = []

        # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
        # to put in modules that are supposed to be activated sequentially
        model = nn.Sequential(normalization)

        i = 0  # increment every time we see a conv
        for layer in self.cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                # The in-place version doesn't play very nicely with the ContentLoss
                # and StyleLoss we insert below. So we replace with out-of-place
                # ones here.
                #Переопределим relu уровень
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

            model.add_module(name, layer)

            if name in self.content_layers_default:
                # add content loss:
                target = model(self.content_img).detach()
                content_loss = ContentLoss(target)
                model.add_module("content_loss_{}".format(i), content_loss)
                content_losses.append(content_loss)

            if name in self.style_layers_default:
                # add style loss:
                target_feature = model(self.style1_img).detach()
                style_loss = StyleLoss(target_feature)
                model.add_module("style_loss_1_{}".format(i), style_loss)
                style_losses_1.append(style_loss)
                
                if self.mode == 2:
                    target_feature = model(self.style2_img).detach()
                    style_loss = StyleLoss(target_feature)
                    model.add_module("style_loss_2_{}".format(i), style_loss)
                    style_losses_2.append(style_loss)

        # now we trim off the layers after the last content and style losses
        #выбрасываем все уровни после последенего styel loss или content loss
        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
                break

        model = model[:(i + 1)]

        return model, style_losses_1,style_losses_2, content_losses