import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.utils.data as data
from torchvision import transforms
import time
import os
import torch.nn.functional as F
from data.CUB_dataset import CubDataset
from model.resnet_STN import resnet_multi_stn101


train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(512),
    transforms.RandomHorizontalFlip(0.5),
    #     transforms.RandomVerticalFlip(0.5),
    transforms.RandomCrop(448),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    #     transforms.Normalize([0.485, 0.465, 0.406], [0.229, 0.224, 0.225])
])
test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(448),
    transforms.CenterCrop(448),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

batch_size = 16

trainset = CubDataset(transform=train_transform)
testset = CubDataset(transform=test_transform, test=True)


trainloader = data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)
testloader = data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def train(model, criterion, optimizer, schedler, epochs, train_log='train_log', test_log='test_log', saved_model='model'):
    best_acc = 0.0
    for epoch in range(epochs):
        begin = time.time()
        logs = open(train_log, 'a')
        model.train()
        running_corrects = 0
        running_loss = 0.0
        schedler.step()
        for i, (images, labels) in enumerate(trainloader):
            start = time.time()
            images = images.cuda()
            labels = labels.cuda()

            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * images.size(0)
            running_corrects += torch.sum(preds == labels.data)

            if i % 10 == 0:
                print('Epoch: {}/{}, Iter: {}/{:.0f}, Loss: {:.4f}, Time: {:.4f}s/batch'
                      .format(epoch, epochs, i, trainset.__len__()/batch_size+1, loss.item(), time.time()-start))
        epoch_loss = running_loss / trainset.__len__()
        epoch_acc = running_corrects.double() / trainset.__len__()

        log = 'Epoch: {}/{}, Loss: {:.4f} Acc: {}/{}, {:.4f}, Time: {:.0f}s'.format(epoch,
                                                                                    epochs,
                                                                                    epoch_loss,
                                                                                    running_corrects,
                                                                                    trainset.__len__(),
                                                                                    epoch_acc,
                                                                                    time.time()-begin)
        print(log)
        logs.write(log+'\n')
        val_acc = validate(model, test_log=test_log)
        if best_acc < val_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), '{}_best.pkl'.format(saved_model))


def validate(model, test_log=''):
    begin = time.time()
    if test_log != '':
        logs = open(test_log, 'a')
    model.eval()
    with torch.no_grad():
        running_corrects = 0
        running_loss = 0.0
        for i, (images, labels) in enumerate(testloader):
            images = images.cuda()
            labels = labels.cuda()

            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * images.size(0)
            running_corrects += torch.sum(preds == labels.data)
        epoch_loss = running_loss / testset.__len__()
        epoch_acc = running_corrects.double() / testset.__len__()
        log = 'Test Loss: {:.4f} Acc: {}/{}, {:.4f}, Time: {:.0f}'.format(epoch_loss,
                                                                          running_corrects,
                                                                          testset.__len__(),
                                                                          epoch_acc,
                                                                          time.time()-begin)
        print(log)
        if test_log != '':
            logs.write(log+'\n')
        return epoch_acc


def main():
    model = resnet_multi_stn101(pretrained=True, num_classes=200, p=0)
    model = model.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-5)
    exp_lr_schedler = lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
    if not os.path.exists('logs'):
        os.makedirs('logs')
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')

    train_log = 'logs/train_resnet_multi975_stn101'
    test_log = 'logs/test_resnet_multi975_stn101'
    saved_model = 'checkpoints/resnet_multi975_stn101'
    train(model, criterion, optimizer, exp_lr_schedler,
          epochs=200, train_log=train_log, test_log=test_log, saved_model=saved_model)


if __name__ == '__main__':
    main()
