# Write your code here :-)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse

from torchvision import datasets, transforms, models
from utils import data_loader, preprocessing, save_checkpoint


def model_training(folder, l_r, hidden, epochs, device):
    train_loader, valid_loader, test_loader = data_loader(folder)

    model = models.vgg13(pretrained = True)

    layer_size = model.classifier[0].in_features

    for params in model.parameters():
        params.requires_grad = False

    print('before classifier')
    classifier = nn.Sequential(nn.Linear(layer_size, hidden),
                               nn.ReLU(),
                               nn.Dropout(p=0.2),
                               nn.Linear(hidden, 102),
                               nn.LogSoftmax(dim=1))

    print('before model.classifier')
    model.classifier = classifier

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=l_r)

    # if device == 'cuda':
    #   model.cuda()
    # else:
    #    model.cpu()
    print("before model.to(device)")
    model.to(device)
    print('after model.to(device)')


    steps = 0
    print_every = 40

    print('for n in epochs')
    for n in range(epochs):
        running_loss = 0
        for inputs, labels in train_loader: #labels are flower names, need to use "flowers" in next loop
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logpred = model.forward(inputs)
            loss = criterion(logpred, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps%print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                print('with torch.no_grad()')
                with torch.no_grad():
                    for inputs, labels in valid_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logpred = model(inputs)
                        batch_loss = criterion(logpred, labels)

                        test_loss += batch_loss.item() #this is tested against the valid data set

                        ps = torch.exp(logpred)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    training_loss = running_loss/print_every
                    avg_validation_loss = test_loss / len(valid_loader)
                    avg_validation_accuracy = accuracy / len(valid_loader)

                print("Epoch: {}/{} | ".format(n+1, epochs),
                      "Training Loss: {:.3f} | ".format(training_loss),
                      "Validation Loss: {:.3f} | ".format(avg_validation_loss),
                      "Validation Accuracy: {:.3f}".format(avg_validation_accuracy))
                running_loss = 0
                model.train()
    return model



def main():
    #Additional arguments
    parsit = argparse.ArgumentParser(description='Training a new network on a data set')

    parsit.add_argument('data_dir', type=str)
    parsit.add_argument('--save_dir', type=str)
    parsit.add_argument('--arch', type=str)
    parsit.add_argument('--learning_rate', type=float)
    parsit.add_argument('--hidden_units', type=int)
    parsit.add_argument('--epochs', type=int)
    parsit.add_argument('--gpu', action = 'store_true')

    args, _ = parsit.parse_known_args()

    folder = args.data_dir

    save_dir = './'
    if args.save_dir:
        save_dir = args.save_dir

    model_arch = 'vgg13'
    if args.arch:
        model_arch = args.arch

    l_r = 0.001
    if args.learning_rate:
        l_r = args.learning_rate

    hidden = 784
    if args.hidden_units:
        hidden = args.hidden_units

    epochs = 5
    if args.epochs:
        epochs = args.epochs

    print('setting device to cuda or cpu...')
    device = 'cuda'
    if args.gpu:
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'

    print('preprocessing started')
    train_data, valid_data, test_data = preprocessing(folder)

    print('data_loader(folder)')
    train_loader, valid_loader, test_loader = data_loader(folder)

    print('model_training(folder,  l_r ... )')
    my_model = model_training(folder, l_r, hidden, epochs, device)

    save_checkpoint(my_model, train_data, save_dir)

    print('Complete')

if __name__ == "__main__":
    main()
