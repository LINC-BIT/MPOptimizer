import torch
import torchvision
import torch.nn as nn
from src.models.resnet import ResNet18
from src.models.resnet_gn import resnet50
from src.DigitModel import load_model
import torch.optim as optim
import torchvision.transforms as transforms
from tqdm import tqdm
import os
import sys
os.chdir(sys.path[0]) 
sys.path.append(os.getcwd()) 
sys.path.append("./..")
import tent as tent 

def main():
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

    net.load_state_dict(torch.load(model_weight_path, map_location=device))
    net=tent.configure_model(net)
    params,param_names=tent.collect_params(net)
    net.to(device)

    optimizer = optim.Adam(params, lr=0.0001)
    tented_model=tent.Tent(net,optimizer)
    epochs = 30
    best_acc = 0.0
    save_path = './weights/digit_M2U.pth'
    train_steps = len(train_loader)
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
        # print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
        #       (epoch + 1, running_loss / train_steps, val_accurate))
        print('[epoch %d]  val_accuracy: %.3f' %
              (epoch + 1,  val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    print('Finished Training')


if __name__ == '__main__':
    main()