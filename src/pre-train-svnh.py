import torch
import torchvision
import torch.nn as nn
from src.models.resnet import ResNet18
from src.models.resnet_gn import resnet50
from src.DigitModel import load_model
from SHOT_loss import CrossEntropyLabelSmooth
from SHOT_loss import SoftmaxEntropyLoss
from itertools import cycle
import torch.optim as optim
import torchvision.transforms as transforms
from tqdm import tqdm
import os
import sys
os.chdir(sys.path[0]) 
sys.path.append(os.getcwd()) 

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    transform = transforms.Compose(
        [
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    transform2 = transforms.Compose(
        [transforms.Lambda(lambda x: x.convert("RGB")),
         transforms.ToTensor(),
         transforms.Resize(32),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_set = torchvision.datasets.SVHN(root='./data', split='train',
                                             download=True, transform=transform)
    
    train_unsupervised_set=torchvision.datasets.MNIST(root='./data', train=True,
                                           download=True, transform=transform2)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=128,
                                               shuffle=True, num_workers=0)
    train_unsupervised_loader = torch.utils.data.DataLoader(train_unsupervised_set, batch_size=128,
                                               shuffle=True, num_workers=0)

    val_set = torchvision.datasets.SVHN(root='./data', split='test',
                                           download=True, transform=transform)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=5000,
                                             shuffle=False, num_workers=0)
    val_data_iter = iter(val_loader)
    val_image, val_label = next(val_data_iter)
    val_num = len(val_set)
    
    #net = resnet50()
    net=load_model('stom').to(device)
    net.to(device)

    # define loss function
    loss_function = nn.CrossEntropyLoss()

    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.0001)
    unsupervised_proportion=0.01
    epochs = 200
    best_acc = 0.0
    save_path = './weights/digit_USPS.pth'
    train_steps = len(train_loader)
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(enumerate(zip(train_loader,cycle(train_unsupervised_loader))), file=sys.stdout)
        for step, data in train_bar:
            images, labels = data[0]
            unsupervised_images,label2=data[1]
            optimizer.zero_grad()
            logits = net(images.to(device))
            outputs=net(unsupervised_images.to(device))
            loss1=unsupervised_proportion*SoftmaxEntropyLoss()(outputs)
            loss2 =CrossEntropyLabelSmooth(num_classes=10, epsilon=0.1)(logits, labels.to(device))
            loss=loss1+loss2
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
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    print('Finished Training')


if __name__ == '__main__':
    main()