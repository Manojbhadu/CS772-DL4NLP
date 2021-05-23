from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import pdb
import tkinter as tk
from tkinter import messagebox
from collections import OrderedDict
from matplotlib import pyplot as plt

model = SentenceTransformer('bert-base-nli-cls-token')


class Concat_embed(nn.Module):
    def __init__(self, embed_dim, projected_embed_dim):
        super(Concat_embed, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(in_features=embed_dim, out_features=projected_embed_dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

    def forward(self, inp, embed):
        projected_embed = self.projection(embed)
        replicated_embed = projected_embed.repeat(
            4, 4, 1, 1).permute(2,  3, 0, 1)
        hidden_concat = torch.cat([inp, replicated_embed], 1)

        return hidden_concat


class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        self.image_size = 64
        self.num_channels = 3
        self.noise_dim = 100
        self.embed_dim = 768
        self.projected_embed_dim = 128
        self.latent_dim = self.noise_dim + self.projected_embed_dim
        self.ngf = 64

        self.projection = nn.Sequential(
            nn.Linear(in_features=self.embed_dim,
                      out_features=self.projected_embed_dim),
            nn.BatchNorm1d(num_features=self.projected_embed_dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.netG = nn.Sequential(
            nn.ConvTranspose2d(self.latent_dim, self.ngf *
                               8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(self.ngf * 8, self.ngf * \
                               4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(self.ngf * 4, self.ngf * \
                               2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(self.ngf * 2, self.ngf,
                               4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(
                self.ngf, self.num_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (num_channels) x 64 x 64
        )

    def forward(self, embed_vector, z):
        projected_embed = self.projection(
            embed_vector).unsqueeze(2).unsqueeze(3)
        latent_vector = torch.cat([projected_embed, z], 1)
        output = self.netG(latent_vector)
        return output


def returnImage(g, sentence):
    encoded = model.encode([sentence])
    encoded = torch.from_numpy(encoded)
    encoded = torch.repeat_interleave(encoded, 64, dim=0)
    noise = torch.randn(64, 100, 1, 1)
    encoded = encoded.reshape(64, 768)
    print(encoded.shape)
    print(noise.shape)
    pred = g(encoded, noise)
    image = pred[0, :, :, :]
    return image.detach().cpu().numpy()


g = generator()
# original saved file with DataParallel
state_dict = torch.load("./Data/checkpoints/gen_54.pth")
# create new OrderedDict that does not contain `module.`
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:]  # remove `module.`
    new_state_dict[name] = v
# load params
g.load_state_dict(new_state_dict)
print(g)


def function():
    result = T.get("1.0", "end")
    if result == "\n":
        messagebox.showinfo('Message', "Enter Sentence")
    else:
        result = result.rstrip("\n")
        image = returnImage(g, result)
        image = np.reshape(image, (64, 64, 3))
        fig = plt.figure("Image Predicted for sentence")
        plt.imshow(image, interpolation='nearest')
        plt.show()


window = tk.Tk()
window.title("Sentiment Analysis")
window.geometry("500x500")

l = tk.Label(window, text="Enter Sentence in context of flowers")
l.config(font=("Courier", 15))
T = tk.Text(window, height=20, width=60)
button = tk.Button(window, text='Generate', width=20, command=function)
l.pack()
T.pack()
button.pack()
window.mainloop()
