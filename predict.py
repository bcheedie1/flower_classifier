# Write your code here :-)
#import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as models
import json
import PIL
import argparse

from torch import nn, tensor, optim
from torchvision import datasets, transforms
from PIL import Image
from utils import load_checkpoint, process_image


def model_prediction(image_path, model, topk, device):
    if torch.cuda.is_available() and device == 'gpu':
        model.to('cuda')

    img = process_image(image_path)
    img = img.unsqueeze_(0)
    img = img.float()

    if device == 'gpu':
        with torch.no_grad():
            output = model.forward(img.cuda())
    else:
        with torch.no_grad():
            output = model.forward(img)

    prob = torch.exp(output)
    top_p, top_class = prob.topk(topk)

    idx_to_class = {val: key for key, val in
                    model.class_to_idx.items()}
    mapped_classes = list()

    for label in top_class.cpu().numpy()[0]:
        mapped_classes.append(idx_to_class[label])

    return top_p, mapped_classes


def main():
    parsit = argparse.ArgumentParser(description='Prediction with probabilities')

    parsit.add_argument('input', default='./flowers/test/1/image_06752.jpg', nargs='?', action="store", type=str)
    parsit.add_argument('--dir', action="store", dest="data_dir", default="./flowers/")
    parsit.add_argument('checkpoint', default='./checkpoint.pth', nargs='?', action="store", type=str)
    parsit.add_argument('--top_k', default=5, dest="top_k", action="store", type=int)
    parsit.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')
    parsit.add_argument('--gpu', default="gpu", action="store", dest="gpu")

    args = parsit.parse_args()
    image_path = args.input
    topk = args.top_k
    device = args.gpu

    chk_path = 'checkpoint.pth'
    if args.checkpoint:
        chk_path = args.checkpoint

    model = load_saved(chk_path)

    with open('cat_to_name.json', 'r') as json_file:
        cat_to_name = json.load(json_file)

    probs, flower_names = model_prediction(image_path, model, topk, device)

    name = flower_names[0]

    labels = []

    for class_idx in flower_names:
        labels.append(cat_to_name[class_idx])

    probs = probs.cpu().numpy()

    print('Most likely:'.format(topk))
    i = 0
    while i < topk:
        print("{} probability of {:.4f}".format(labels[i], probs[0][i]))
        i += 1
    print("Complete")


if __name__ == "__main__":
    main()
