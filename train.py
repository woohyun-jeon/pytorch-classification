import os
import random
import argparse
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

import models

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder


def get_arguments():
    parser = argparse.ArgumentParser(description='parameters for image classification model using ImageNet dataset')
    parser.add_argument('--model_name', type=str, default='vgg11', help='model name')
    parser.add_argument('--data_dir', type=str, default='E:/test/classification/data', help='dataset path')
    parser.add_argument('--output_dir', type=str, default='E:/test/classification/outputs', help='output path')
    parser.add_argument('--epochs', type=int, default=500, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers')
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')

    args = parser.parse_args()

    return args


def train(model, device, train_dataloader, optimizer, criterion):
    model.train()

    train_loss = 0.
    for batch_idx, (data, target) in tqdm(enumerate(train_dataloader), desc='-- Training --', total=len(train_dataloader)):
        data, target = data.to(device), target.to(device)

        # feed forward
        output = model(data)

        # get loss
        loss = criterion(output, target)
        train_loss += loss.item()

        # back propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= (batch_idx+1)

    return train_loss


def valid(model, device, valid_dataloader, criterion):
    model.eval()

    valid_loss = 0.
    total = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in tqdm(enumerate(valid_dataloader), desc='-- Validation --', total=len(valid_dataloader)):
            data, target = data.to(device), target.to(device)

            # get predicted
            output = model(data)
            _, preds = torch.max(output, 1)
            total += target.size(0)
            correct += (preds == target).sum().item()

            # get loss
            loss = criterion(output, target)
            valid_loss += loss.item()

    valid_loss /= (batch_idx+1)
    valid_acc = correct / total

    return valid_loss, valid_acc


def main():
    # get arguments
    args = get_arguments()

    # confirm device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # fix seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # generate dataset
    transform_train = transforms.Compose([
        transforms.Resize(size=(224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    transform_valid = transforms.Compose([
        transforms.Resize(size=(224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = ImageFolder(root=os.path.join(args.data_dir, 'train'), transform=transform_train)
    valid_dataset = ImageFolder(root=os.path.join(args.data_dir, 'val'), transform=transform_valid)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # set model
    model = models.__dict__[args.model_name](in_channels=3, num_classes=len(train_dataset.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=200)

    # train
    train_losses = []
    valid_losses = []
    best_acc = 0.
    acc_patience = 0
    for epoch in range(args.epochs):
        print('[Start] Epoch {}/{}'.format(epoch + 1, args.epochs))
        train_loss = train(model, device, train_dataloader, optimizer, criterion)
        valid_loss, valid_acc = valid(model, device, valid_dataloader, criterion)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        print('epoch: {}/{} -- '.format(epoch+1, args.epochs),
              'train loss: {:.4f}'.format(train_loss/len(train_dataloader)),
              'val loss: {:.4f}'.format(valid_loss/len(valid_dataloader)),
              'val acc: {:.4f}'.format(valid_acc*100)
              )

        print('[Complete] Epoch {}/{}'.format(epoch + 1, args.epochs))

        scheduler.step()

        if abs(valid_acc - best_acc) < 0.0001:
            acc_patience += 1
            if acc_patience == 10:
                print('epoch {} -- early stopping'.format(epoch))
                break

        if valid_acc > best_acc:
            best_acc = valid_acc
            torch.save(model.state_dict(), os.path.join(args.output_dir, args.model_name + '_imagenet_224.pth'))


    # save loss changes to image file
    plt.figure(figsize=(7.5, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='train')
    plt.plot(range(1, len(valid_losses) + 1), valid_losses, label='valid')
    plt.title('Train-Valid Loss', fontsize=20)
    plt.xlabel('Epochs', fontsize=15)
    plt.ylabel('Loss', fontsize=15)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(fontsize=10)
    plt.savefig(os.path.join(args.output_dir, args.model_name + '_loss.png'))


if __name__ == '__main__':
    main()