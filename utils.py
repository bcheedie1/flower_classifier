# Write your code here :-)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


from torchvision import datasets, transforms, models
from PIL import Image

def preprocessing(folder):
    data_dir = folder
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # TODO: Define your transforms for the training, validation, and testing sets
    #data_transforms =

    train_transforms = transforms.Compose([transforms.RandomRotation(90),
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.CenterCrop(224),
                                          transforms.Resize(255),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # TODO: Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform = train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform = valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform = test_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size = 40, shuffle = True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size = 40)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size = 40)

    return train_data, valid_data, test_data


def data_loader(folder):
    data_dir = folder

    train_data, valid_data, test_data = preprocessing(data_dir)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size = 40, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size = 40)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size = 40)

    return train_loader, valid_loader, test_loader


def save_checkpoint(model, train_data, save_dir):
    filename = 'checkpoint.pth'
    file = save_dir + filename

    model.class_to_idx = train_data.class_to_idx
    model_state = {'state_dict': model.state_dict(),
                   'classifer': model.classifier,
                   'class_to_idx': model.class_to_idx,
                  }

    torch.save(model_state, file)

    return filename


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = models.vgg13(pretrained = True)
    model.state_dict = checkpoint['state_dict']
    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']

    return model

#saved_model = load_checkpoint('checkpoint.pth')
#saved_model


def process_image(image):

    # TODO: Process a PIL image for use in a PyTorch model
    mean = [0.485, 0.456, 0.406]
    stdev = [0.229, 0.224, 0.225]

    img = Image.open(image)
    w, h = img.size
    if w < h:
        new_size = [256, 100000]
    else:
        new_size = [10000, 256]

    img.thumbnail(size = new_size)

    w, h = img.size
    left = (w - 224) / 2
    right = (w + 224) / 2
    top = (h - 224) / 2
    bottom = (h + 224) / 2
    img = img.crop((left, right, top, bottom))

    np_img = np.array(img) / 255
    np_img = (np_img - mean) / stdev

    np_img = np_img.transpose(2, 0, 1)

    return np_img
