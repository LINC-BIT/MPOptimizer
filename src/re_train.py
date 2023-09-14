import os
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import logging
from torchvision import transforms, datasets
from tqdm import tqdm
from dataset import CIFAR10C
from torch.utils.data import DataLoader
from src.models.resnet import ResNet50
from src.models.resnet_gn import resnext50_32x4d
from utils import load_txt
from dataset import CIFAR10C
import sys
os.chdir(sys.path[0]) 
sys.path.append(os.getcwd()) 
sys.path.append("./..")
import tent_gn as tent

logger = logging.getLogger(__name__)

CORRUPTIONS = load_txt('./corruptions.txt')
MEAN = [0.49139968, 0.48215841, 0.44653091]
STD  = [0.24703223, 0.24348513, 0.26158784]

def retrain(microbatch_size,checkpoint_size,freezing_rate,t):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    transform = transforms.Compose(
        [
         transforms.ToTensor(),
         transforms.Resize(28),
         transforms.Normalize((0.5), (0.5))])

    train_set = torchvision.datasets.SVHN(root='./data', split='train',
                                             download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=128,
                                               shuffle=True, num_workers=0)

    val_set = torchvision.datasets.USPS(root='./data', train=False,
                                           download=True, transform=transform)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=2,
                                             shuffle=False, num_workers=0)
    val_data_iter = iter(val_loader)
    val_image, val_label = next(val_data_iter)
    val_num = len(val_set)
    
    #net = resnet50()
    net=load_model('utom').to(device)
    model_weight_path = "./weights/digit_M2U.pth"

    if freezing_rate==true:
        net.load_state_dict(torch.load(model_weight_path, map_location=device))
        net=tent.configure_model(net)
        params,param_names=tent.collect_params(net)
        net.to(device)
        accum_steps=256/microbatch_size

        optimizer = optim.Adam(params, lr=0.0001)
        tented_model=tent.Tent(net,optimizer,accum_steps)
        epochs = 30
        best_acc = 0.0
        save_path = './weights/digit_M2U.pth'
        train_steps = len(train_loader)
        budget=t
        while budget>0:
            since = time.time()
            for epoch in range(epochs):
                # validate
                net.eval()
                acc = 0.0  # accumulate accurate number / epoch
                with torch.no_grad():
                    val_bar = tqdm(val_loader, file=sys.stdout)
                    for val_data in val_bar:
                        val_images, val_labels = val_data
                        outputs = tented_model(val_images.to(device))
                        # loss = loss_function(outputs, test_labels)
                        predict_y = torch.max(outputs, dim=1)[1]
                        acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

                        val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                                epochs)

                val_accurate = acc / val_num
                print('[epoch %d]  val_accuracy: %.3f' %
                    (epoch + 1,  val_accurate))

                if val_accurate > best_acc:
                    best_acc = val_accurate
                    torch.save(net.state_dict(), save_path)
            
            time_elapsed = time.time() - since
            budget=budget-time_elapsed



    print('Finished Training')

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)])
        
    net = resnext50_32x4d()

    model_weight_path = "./weights/resnext50_c.pth"

    net.load_state_dict(torch.load(model_weight_path, map_location=device))

    # in_channel = net.fc.in_features
    # net.fc = nn.Linear(in_channel, 10)
    net.state_dict()
    net=tent.configure_model(net)
    params,param_names=tent.collect_params(net)
    net.to(device)
    optimizer = optim.Adam(params, lr=0.0001)
    optimizer_tent=optim.Adam(params, lr=0.00001)
    tented_model=tent.Tent(net,optimizer_tent)
    loss_function = nn.CrossEntropyLoss()             
    #accum_steps=1
    # net=tent.configure_model(net)
    # params,param_names=tent.collect_params(net)
    # net.to(device)
    # optimizer = optim.Adam(params, lr=0.00001)
    # tented_model=tent.Tent(net,optimizer)
    # logger.info(f"model for adaptation: %s", net)
    # logger.info(f"params for adaptation: %s", param_names)
    # logger.info(f"optimizer for adaptation: %s", optimizer)
    with tqdm(total=len(CORRUPTIONS), ncols=80) as pbar:
        for ci, cname in enumerate(CORRUPTIONS):
            # load dataset
            if cname == 'natural':
                dataset = datasets.CIFAR10(
                    os.path.join("../data_set/CIFAR-10-C", 'cifar10'),
                    train=False, transform=transform, download=True,
                )
            else:
                dataset = CIFAR10C(
                    os.path.join("../data_set/CIFAR-10-C", 'cifar10-c'),
                    cname, transform=transform
                )
            #tented_model.reset()
            train_size = int(len(dataset) * 0.3)
            test_size = len(dataset) - train_size
            train_set, val_set = torch.utils.data.random_split(dataset, [train_size, test_size])
            train_loader = torch.utils.data.DataLoader(train_set, batch_size=128,shuffle=True, num_workers=4)
            val_loader=torch.utils.data.DataLoader(val_set, batch_size=8,shuffle=False, num_workers=4)
            # val_data_iter = iter(val_loader)
            # val_image, val_label = next(val_data_iter)
            val_num = len(val_set)
            best_acc = 0
            save_path = './weights/resnext50_t.pth'
            train_steps = len(train_loader)

            train_bar = tqdm(train_loader, file=sys.stdout)
            net.train()
            running_loss = 0.0
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
                    loss = loss_function(logits, labels.to(device))
                    loss.backward()
                    optimizer.step()

            # print statistics
                    running_loss += loss.item()

                    train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)
        # validate
            tented_model.eval()
            acc = 0.0  # accumulate accurate number / epoch
            with torch.no_grad():
                val_bar = tqdm(val_loader, file=sys.stdout)
                for val_data in val_bar:
                    val_images, val_labels = val_data
                    outputs = tented_model(val_images.to(device))
                    # loss = loss_function(outputs, test_labels)
                    predict_y = torch.max(outputs, dim=1)[1]
                    acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

                    val_bar.desc = "valid epoch[{}/{}]".format(1,1)

                val_accurate = acc / val_num
                print('[epoch %d]  val_accuracy: %.3f' %(1,val_accurate))

            if val_accurate > best_acc:
                best_acc = val_accurate
                torch.save(net.state_dict(), save_path)

        print('Finished Training')

if __name__ == '__main__':
    main()
