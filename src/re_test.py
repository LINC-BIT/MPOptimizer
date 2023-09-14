import os
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets
from tqdm import tqdm
from dataset import CIFAR10C
from SHOT_loss import CrossEntropyLabelSmooth
from SHOT_loss import SoftmaxEntropyLoss
from torch.utils.data import DataLoader
from src.models.resnet import ResNet18
from src.models.resnet_gn import resnet50
from src.models.mobilevit import mobile_vit_xx_small
from utils import load_txt
from dataset import CIFAR10C
import sys
os.chdir(sys.path[0]) 
sys.path.append(os.getcwd()) 
sys.path.append("./..")
import tent as tent

CORRUPTIONS = load_txt('./corruptions.txt')
MEAN = [0.49139968, 0.48215841, 0.44653091]
STD  = [0.24703223, 0.24348513, 0.26158784]

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)])
        
    net = mobile_vit_xx_small()

    model_weight_path = "./weights/mobilevit_t.pth"

    # net.load_state_dict(torch.load(model_weight_path, map_location=device))

    # # in_channel = net.classifier.fc.in_features
    # # net.classifier.fc = nn.Linear(in_channel, 1000)
    # net.state_dict()

    pre_dic=torch.load(model_weight_path, map_location=device)
    model_dic=net.state_dict()
    pre_dic={k: v for k, v in pre_dic.items() if (k in model_dic and 'fc' not in k)}
    model_dic.update(pre_dic)
    net.load_state_dict(model_dic)

    accum_steps=1
    net.to(device)
    loss_function = nn.CrossEntropyLoss()

    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.00001)

    with tqdm(total=len(CORRUPTIONS), ncols=80) as pbar:
        for ci, cname in enumerate(CORRUPTIONS):
            # load dataset
            if cname == 'natural':
                dataset = datasets.CIFAR10(
                    os.path.join("../data_set/CIFAR-10-C", 'cifar10'),
                    train=True, transform=transform, download=True,
                )
            else:
                dataset = CIFAR10C(
                    os.path.join("../data_set/CIFAR-10-C", 'cifar10-c'),
                    cname, transform=transform
                )
            train_size = int(len(dataset) * 0.8)
            test_size = len(dataset) - train_size
            train_set, val_set = torch.utils.data.random_split(dataset, [train_size, test_size])
            train_loader = torch.utils.data.DataLoader(train_set, batch_size=128,shuffle=True, num_workers=4)
            val_loader=torch.utils.data.DataLoader(val_set, batch_size=5000,shuffle=False, num_workers=4)
            #val_data_iter = iter(val_loader)
            #val_image, val_label = next(val_data_iter)
            val_num = len(val_set)
            best_acc = 0.0
            save_path = './weights/mobilevit_recl.pth'
            train_steps = len(train_loader)
            unsupervised_proportion=0.01
            epochs = 10
            for epoch in range(epochs):
        # train
                net.train()
                running_loss = 0.0
                train_bar = tqdm(train_loader, file=sys.stdout)
                for step, data in enumerate(train_bar):
                    images, labels = data                               
                    optimizer.zero_grad()
                    logits = net(images.to(device))
                    loss1=unsupervised_proportion*SoftmaxEntropyLoss()(logits)
                    # loss2 = CrossEntropyLabelSmooth(num_classes=10, epsilon=0.1)(logits, labels.to(device))
                    loss2=torch.nn.CrossEntropyLoss(label_smoothing=0.1)(logits, labels.to(device))
                    loss=loss1+loss2
                    # loss=torch.nn.CrossEntropyLoss(label_smoothing=0.1)(logits, labels.to(device))
                    loss.backward()
                    optimizer.step()

            # print statistics
                    running_loss += loss.item()

                    train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)

        # validate
                net.eval()
                acc = 0.0  # accumulate accurate number / epoch
                with torch.no_grad():
                    val_bar = tqdm(val_loader, file=sys.stdout)
                    for val_data in val_bar:
                        val_images, val_labels = val_data
                        outputs = net(val_images.to(device))
                    # loss = loss_function(outputs, test_labels)
                        predict_y = torch.max(outputs, dim=1)[1]
                        acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

                        val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                            epochs)

                val_accurate = acc / val_num
                print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %(epoch + 1, running_loss / train_steps, val_accurate))

                if val_accurate > best_acc:
                    best_acc = val_accurate
                    torch.save(net.state_dict(), save_path)

            print('Finished Training')

if __name__ == '__main__':
    main()
