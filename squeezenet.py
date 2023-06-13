import random

from pandas.core.common import flatten

import optuna

import torch
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

import glob

train_transforms = A.Compose(
    [
        A.SmallestMaxSize(max_size=350),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=360, p=0.5),
        A.RandomCrop(height=256, width=256),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.MultiplicativeNoise(multiplier=[0.5,2], per_channel=True, p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
        ToTensorV2(),
    ]
)

test_transforms = A.Compose(
    [
        A.SmallestMaxSize(max_size=350),
        A.CenterCrop(height=256, width=256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)


train_data_path = './data/grayscale/train'
test_data_path = './data/grayscale/test'

train_image_paths = []
classes = []

for data_path in glob.glob(train_data_path + '/*'):
    classes.append(data_path.split('/')[-1])
    train_image_paths.append(glob.glob(data_path + '/*'))

train_image_paths = list(flatten(train_image_paths))
random.shuffle(train_image_paths)

print('train_image_path example: ', train_image_paths[0])
print('class example: ', classes[0])

test_image_paths = []
for data_path in glob.glob(test_data_path + '/*'):
    test_image_paths.append(glob.glob(data_path + '/*'))

test_image_paths = list(flatten(test_image_paths))

print("Train size: {}\nTest size: {}".format(len(train_image_paths), len(test_image_paths)))


idx_to_class = {i:j for i, j in enumerate(classes)}
class_to_idx = {value:key for key,value in idx_to_class.items()}


class TomatoGrayscaleDataset(Dataset):
    def __init__(self, image_paths, transform=False):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_filepath = self.image_paths[idx]
        image = cv2.imread(image_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = image_filepath.split('/')[-2]
        label = class_to_idx[label]
        if self.transform is not None:
            image = self.transform(image=image)["image"]

        return image, label


#######################################################
#                  Create Dataset
#######################################################

train_dataset = TomatoGrayscaleDataset(train_image_paths, train_transforms)
test_dataset = TomatoGrayscaleDataset(test_image_paths, test_transforms)


def get_data_loaders(batch_size):
    trainloader = DataLoader(train_dataset, batch_size=batch_size,
                             num_workers=2, shuffle=True)

    testloader = DataLoader(test_dataset, batch_size=batch_size,
                            num_workers=2)

    return trainloader, testloader


def train(model, dataloader, optimizer, criterion, device):
    model.train()
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()


def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
    test_loss = running_loss / len(dataloader.dataset)
    test_acc = running_corrects.double() / len(dataloader.dataset)
    return test_loss, test_acc.item()


def objective(trial):
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    num_epochs = trial.suggest_int('num_epochs', 10, 100)
    dropout_rate = trial.suggest_uniform('dropout_rate', 0, 0.5)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-10, 1e-3)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.squeezenet1_1(pretrained=True)
    model.classifier[1] = torch.nn.Conv2d(512, 10, kernel_size=(1, 1), stride=(1, 1))
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    trainloader, testloader = get_data_loaders(batch_size)

    for epoch in range(num_epochs):
        train(model, trainloader, optimizer, criterion, device)
        test_loss, test_acc = evaluate(model, testloader, criterion, device)
        trial.report(test_acc, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return test_acc


# study = optuna.create_study(direction='maximize')
# study.optimize(objective, n_trials=50)
#
#
# print("Best trial:")
# best_trial = study.best_trial
# print(f"  Value: {best_trial.value}")
# print("  Params: ")
# for key, value in best_trial.params.items():
#     print(f"    {key}: {value}")
#
#
# best_lr = best_trial.params['lr']
# best_batch_size = best_trial.params['batch_size']
# best_num_epochs = best_trial.params['num_epochs']
# best_dropout_rate = best_trial.params['dropout_rate']
# best_weight_decay = best_trial.params['weight_decay']
#
#
# model = models.squeezenet1_1(pretrained=True)
# model.classifier[1] = torch.nn.Conv2d(512, 10, kernel_size=(1,1), stride=(1,1))
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = model.to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=best_lr, weight_decay=best_weight_decay)
# criterion = torch.nn.CrossEntropyLoss()
# trainloader, testloader = get_data_loaders(best_batch_size)
#
#
# for epoch in range(best_num_epochs):
#     train(model, trainloader, optimizer, criterion, device)
#     test_loss, test_acc = evaluate(model, testloader, criterion, device)
#     print(f"Epoch {epoch+1}: Test accuracy = {test_acc:.4f}")

if __name__ == '__main__':

    model = models.squeezenet1_1(pretrained=True)
    model.classifier[1] = torch.nn.Conv2d(512, 10, kernel_size=(1, 1), stride=(1, 1))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()
    trainloader, testloader = get_data_loaders(16)

    for epoch in range(60):
        train(model, trainloader, optimizer, criterion, device)
        test_loss, test_acc = evaluate(model, testloader, criterion, device)
        print(f"Epoch {epoch + 1}: Test accuracy = {test_acc:.4f}")

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)

    print("Best trial:")
    best_trial = study.best_trial
    print(f"  Value: {best_trial.value}")
    print("  Params: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

    best_lr = best_trial.params['lr']
    best_batch_size = best_trial.params['batch_size']
    best_num_epochs = best_trial.params['num_epochs']
    best_dropout_rate = best_trial.params['dropout_rate']
    best_weight_decay = best_trial.params['weight_decay']

    model = models.squeezenet1_1(pretrained=True)
    model.classifier[1] = torch.nn.Conv2d(512, 10, kernel_size=(1, 1), stride=(1, 1))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=best_lr, weight_decay=best_weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    trainloader, testloader = get_data_loaders(best_batch_size)

    for epoch in range(best_num_epochs):
        train(model, trainloader, optimizer, criterion, device)
        test_loss, test_acc = evaluate(model, testloader, criterion, device)
        print(f"Epoch {epoch + 1}: Test accuracy = {test_acc:.4f}")


