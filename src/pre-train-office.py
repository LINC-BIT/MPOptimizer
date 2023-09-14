import torch
import torchvision
import json
import torch.nn as nn
from src.models.resnet import ResNet50
from src.models.resnet_gn import resnet50
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
import os
from SHOT_loss import CrossEntropyLabelSmooth
from SHOT_loss import SoftmaxEntropyLoss
from itertools import cycle
import sys
os.chdir(sys.path[0]) 
sys.path.append(os.getcwd()) 

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    transform = transforms.Compose(
        [transforms.Resize([256, 256]),
         transforms.RandomCrop(224),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor()])

    data_root = os.path.abspath(os.path.join(os.getcwd(), "./.."))  # get data root path
    image_path = os.path.join(data_root, "data_set", "office31")  # office data set path
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    dataset = datasets.ImageFolder(root=os.path.join(image_path, "amazon"),
                                         transform=transform)
    unsupervised_trainset = datasets.ImageFolder(root=os.path.join(image_path, "webcam"),
                                         transform=transform)
    train_num = len(dataset)

 
    office_list = dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in office_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=30)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 32
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_size = int(len(dataset) * 0.7)
    test_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=32,shuffle=True, num_workers=4)
    train_unsupervised_loader = torch.utils.data.DataLoader(unsupervised_trainset, batch_size=128,shuffle=True, num_workers=4)
    val_loader=torch.utils.data.DataLoader(val_set, batch_size=32,shuffle=False, num_workers=4)
    val_num = len(val_set)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))
    
    net = resnet50()
    # load pretrain weights
    # download url: https://download.pytorch.org/models/resnet34-333f7ec4.pth
    # model_weight_path = "./resnet56-pre.pth"
    # assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    # net.load_state_dict(torch.load(model_weight_path, map_location='cpu'))
    # # for param in net.parameters():
    # #     param.requires_grad = False

    # # change fc layer structure
    # in_channel = net.fc.in_features
    # net.fc = nn.Linear(in_channel, 10)
    net.to(device)

    # define loss function
    #loss_function = nn.CrossEntropyLoss()

    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.0001)
    unsupervised_proportion=0.01
    epochs = 100
    best_acc = 0.0
    save_path = './weights/resnet50_office_gn_smooth.pth'
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
            loss2 =CrossEntropyLabelSmooth(num_classes=31, epsilon=0.1)(logits, labels.to(device))
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