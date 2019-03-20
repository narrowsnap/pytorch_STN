import torch.utils.data as data
import os
import cv2


class CubDataset(data.Dataset):
    def __init__(self,
                 root='CUB_200_2011/',
                 transform=None,
                 test=False
                 ):
        self.root = root
        self.transform = transform
        self.x = []
        self.y = []
        self.load_data_list(test)

    def load_data_list(self, test):
        images = open(os.path.join(self.root, 'images.txt')).readlines()
        labels = open(os.path.join(self.root, 'image_class_labels.txt')).readlines()
        with open(os.path.join(self.root, 'train_test_split.txt')) as f:
            for line in f.readlines():
                lines = line.strip().split(' ')
                if test and lines[1] == '0':
                    self.x.append(os.path.join(self.root, 'images', images[int(lines[0])-1].strip().split(' ')[1]))
                    self.y.append(int(labels[int(lines[0])-1].strip().split(' ')[1])-1)
                if not test and lines[1] == '1':
                    self.x.append(os.path.join(self.root, 'images', images[int(lines[0])-1].strip().split(' ')[1]))
                    self.y.append(int(labels[int(lines[0])-1].strip().split(' ')[1])-1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        filepath = self.x[item]
        label = self.y[item]

        image = cv2.imread(filepath)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            image = self.transform(image)

        return image, label
