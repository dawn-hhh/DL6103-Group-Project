import torch
import torchvision
import torch.nn as nn
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.models import resnet50
from cutmix.cutmix import CutMix
from cutmix.utils import CutMixCrossEntropyLoss
from tqdm import tqdm
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def cifar100_dataset(data_augmentation=True):
    if data_augmentation:
        transform_train = transforms.Compose([
            transforms.Resize((32,32)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5070,0.4865,0.4409],[0.2627,0.2564,0.2761])
        ])
        transform_test=transforms.Compose(
            [transforms.Resize((32,32)),
             transforms.ToTensor(),
             transforms.Normalize([0.5070,0.4865,0.4409],[0.2627,0.2564,0.2761])])
    else:
        transform_train = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])
        transform_test = transforms.Compose(
            [transforms.Resize((32, 32)),
             transforms.ToTensor()])

    cifar100_training=torchvision.datasets.CIFAR100(root="./data", train=True, download=True, transform=transform_train)
    trainloader=torch.utils.data.DataLoader(cifar100_training, batch_size=128, shuffle=True)

    cifar100_testing=torchvision.datasets.CIFAR100(root="./data", train=False, download=True, transform=transform_test)
    testloader=torch.utils.data.DataLoader(cifar100_testing, batch_size=128, shuffle=False)

    return trainloader,testloader


def Conv1(in_planes, places, stride=2):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_planes,out_channels=places,kernel_size=7,stride=stride,padding=3, bias=False),
        nn.BatchNorm2d(places),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )

class BasicBlock(nn.Module):
    def __init__(self,in_places,places, stride=1,downsampling=False, expansion = 1):
        super(BasicBlock,self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling

        # torch.Size([1, 64, 56, 56]), stride = 1
        # torch.Size([1, 128, 28, 28]), stride = 2
        # torch.Size([1, 256, 14, 14]), stride = 2
        # torch.Size([1, 512, 7, 7]), stride = 2
        self.basicblock = nn.Sequential(
            nn.Conv2d(in_channels=in_places, out_channels=places, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(places * self.expansion),
        )

        # torch.Size([1, 64, 56, 56])
        # torch.Size([1, 128, 28, 28])
        # torch.Size([1, 256, 14, 14])
        # torch.Size([1, 512, 7, 7])
        if self.downsampling:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_places, out_channels=places*self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(places*self.expansion)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.basicblock(x)

        if self.downsampling:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    def __init__(self,in_places,places, stride=1,downsampling=False, expansion = 4):
        super(Bottleneck,self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling

        self.bottleneck = nn.Sequential(
            # torch.Size([1, 64, 56, 56])，stride=1
            # torch.Size([1, 128, 56, 56])，stride=1
            # torch.Size([1, 256, 28, 28]), stride=1
            # torch.Size([1, 512, 14, 14]), stride=1
            nn.Conv2d(in_channels=in_places,out_channels=places,kernel_size=1,stride=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            # torch.Size([1, 64, 56, 56])，stride=1
            # torch.Size([1, 128, 28, 28]), stride=2
            # torch.Size([1, 256, 14, 14]), stride=2
            # torch.Size([1, 512, 7, 7]), stride=2
            nn.Conv2d(in_channels=places, out_channels=places, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            # torch.Size([1, 256, 56, 56])，stride=1
            # torch.Size([1, 512, 28, 28]), stride=1
            # torch.Size([1, 1024, 14, 14]), stride=1
            # torch.Size([1, 2048, 7, 7]), stride=1
            nn.Conv2d(in_channels=places, out_channels=places * self.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(places * self.expansion),
        )

        # torch.Size([1, 256, 56, 56])
        # torch.Size([1, 512, 28, 28])
        # torch.Size([1, 1024, 14, 14])
        # torch.Size([1, 2048, 7, 7])
        if self.downsampling:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_places, out_channels=places*self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(places*self.expansion)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.bottleneck(x)
        if self.downsampling:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,blocks, blockkinds, num_classes=100):
        super(ResNet,self).__init__()

        self.blockkinds = blockkinds
        self.conv1 = Conv1(in_planes=3, places=64)
        if self.blockkinds == BasicBlock:
            self.expansion = 1
            # 64 -> 64
            self.layer1 = self.make_layer(in_places=64, places=64, block=blocks[0], stride=1)
            # 64 -> 128
            self.layer2 = self.make_layer(in_places=64, places=128, block=blocks[1], stride=2)
            # 128 -> 256
            self.layer3 = self.make_layer(in_places=128, places=256, block=blocks[2], stride=2)
            # 256 -> 512
            self.layer4 = self.make_layer(in_places=256, places=512, block=blocks[3], stride=2)

            self.fc = nn.Linear(512, num_classes)

        if self.blockkinds == Bottleneck:
            self.expansion = 4
            # 64 -> 64
            self.layer1 = self.make_layer(in_places = 64, places= 64, block=blocks[0], stride=1)
            # 256 -> 128
            self.layer2 = self.make_layer(in_places = 256,places=128, block=blocks[1], stride=2)
            # 512 -> 256
            self.layer3 = self.make_layer(in_places=512,places=256, block=blocks[2], stride=2)
            # 1024 -> 512
            self.layer4 = self.make_layer(in_places=1024,places=512, block=blocks[3], stride=2)

            self.fc = nn.Linear(8192, num_classes)

        self.avgpool = nn.AvgPool2d(7, stride=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_layer(self, in_places, places, block, stride):

        layers = []

        # torch.Size([1, 64, 56, 56])  -> torch.Size([1, 256, 56, 56])， stride=1
        # torch.Size([1, 256, 56, 56]) -> torch.Size([1, 512, 28, 28])， stride=2 
        # torch.Size([1, 512, 28, 28]) -> torch.Size([1, 1024, 14, 14])，stride=2 
        # torch.Size([1, 1024, 14, 14]) -> torch.Size([1, 2048, 7, 7])， stride=2
        layers.append(self.blockkinds(in_places, places, stride, downsampling =True))

        # torch.Size([1, 256, 56, 56]) -> torch.Size([1, 256, 56, 56])
        # torch.Size([1, 512, 28, 28]) -> torch.Size([1, 512, 28, 28])
        # torch.Size([1, 1024, 14, 14]) -> torch.Size([1, 1024, 14, 14])
        # torch.Size([1, 2048, 7, 7]) -> torch.Size([1, 2048, 7, 7])
        # print("places*self.expansion:", places*self.expansion)
        # print("block:", block)
        for i in range(1, block):
            layers.append(self.blockkinds(places*self.expansion, places))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)   # torch.Size([1, 64, 56, 56])
        x = self.layer1(x)  # torch.Size([1, 256, 56, 56])
        x = self.layer2(x)  # torch.Size([1, 512, 28, 28])
        x = self.layer3(x)  # torch.Size([1, 1024, 14, 14])
        x = self.layer4(x)  # torch.Size([1, 2048, 7, 7])
        x = self.avgpool(x) # torch.Size([1, 2048, 1, 1]) / torch.Size([1, 512])
        x = x.view(x.size(0), -1)   # torch.Size([1, 2048]) / torch.Size([1, 512])

        x = self.fc(x)      # torch.Size([1, 600])

        return x


def ResNet50():
    return ResNet([3, 4, 6, 3], Bottleneck)


def train(data_augmentation,optimizer_name="sgd",cos=True,weight_decay=False):
    device=torch.device("cuda")
    model=resnet50(pretrained=False)
    model.fc=nn.Linear(2048,100)
    model=model.to(device)
    if optimizer_name=="sgd":
        if weight_decay:
            optimizer=optim.SGD(model.parameters(),lr=1e-2,momentum=0.9,weight_decay=0.0005)
        else:
            optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
    else:
        optimizer=optim.Adam(model.parameters(), lr=1e-2)
    if cos:
        lr_sh=optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=100)
    loss_func=nn.CrossEntropyLoss()
    train_loader,val_loader=cifar100_dataset(data_augmentation=data_augmentation)
    train_loss_epoch,val_loss_epoch=[],[]
    train_acc_epoch,val_acc_epoch=[],[]
    for epoch in range(101):
        model.train()
        data=tqdm(train_loader)
        train_loss_batch=[]
        train_acc_batch=[]
        for batch,(x,y) in enumerate(data):
            datasets_train,labels_train=x.to(device),y.to(device)
            labels_prob=model(datasets_train)
            labels_pred=torch.argmax(labels_prob,dim=-1).cpu().numpy()
            train_acc=np.mean((labels_train.cpu().numpy()==labels_pred).astype(int))
            train_acc_batch.append(train_acc)
            loss_train=loss_func(labels_prob,labels_train)
            train_loss_batch.append(loss_train.item())
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()
            data.set_postfix_str(f"epoch:{epoch+1},batch:{batch+1},loss:{loss_train.item()},train_acc:{train_acc}")

        train_acc_epoch.append(np.mean(train_acc_batch))
        if np.mean(train_loss_batch)<10:
            train_loss_epoch.append(np.mean(train_loss_batch))
        print("start validation!")
        val_loss,val_acc=get_val_result(model,val_loader)
        print(f"val_loss:{val_loss},val_accuracy:{val_acc}")
        val_acc_epoch.append(val_acc)
        if val_loss<10:
            val_loss_epoch.append(val_loss)
        if cos:
            lr_sh.step()
        if (epoch+1) % 100 == 0:
            plot_accuracy_loss(train_acc_epoch, val_acc_epoch,train_loss_epoch, val_loss_epoch,name=f"augmentation({data_augmentation})_optimizer({optimizer_name})_cos({cos})_delay({weight_decay})")
            torch.save(model,f"resnet50_augmentation({data_augmentation})_optimizer({optimizer_name})_cos({cos})_delay({weight_decay})_epoch_{epoch+1}.pth")


def get_val_result(model,val_loader):
    device=torch.device("cuda")
    model.eval()
    labels_true,labels_pred=np.array([]),np.array([])
    loss_func=nn.CrossEntropyLoss()
    losses=[]
    with torch.no_grad():
        for x,y in val_loader:
            datasets_val,labels_val=x.to(device),y.to(device)
            labels_prob=model(datasets_val)
            loss_test=loss_func(labels_prob,labels_val)
            labels_pred=np.concatenate([labels_pred,torch.argmax(labels_prob,dim=-1).cpu().numpy()])
            losses.append(loss_test.item())
            labels_true=np.concatenate([labels_true,labels_val.cpu().numpy()],axis=-1)

    return np.mean(losses),np.mean(np.array(labels_true==labels_pred).astype(int))
    
#dataset = datasets.CIFAR100(args.cifarpath, train=True, download=True, transform=transform_train)
#dataset = CutMix(dataset, num_class=100, beta=1.0, prob=0.5, num_mix=2)
#criterion = CutMixCrossEntropyLoss(True)
#for _ in range(num_epoch):
 #   for input, target in loader:
  #      output = model(input)
   #     loss = criterion(output, target)
    #    loss.backward()
     #   optimizer.step()
      #  optimizer.zero_grad()

def plot_accuracy_loss(train_acc,val_acc,train_loss,val_loss,name):
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.plot(range(1,len(train_acc)+1),train_acc,label="train_acc",c="red")
    plt.plot(range(1,len(val_acc)+1),val_acc,label="val_acc",c="green")
    plt.title("epoch-accuracy")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.subplot(1,2,2)
    plt.plot(range(1, len(train_loss) + 1), train_loss, label="train_loss", c="red")
    plt.plot(range(1, len(val_loss) + 1), val_loss, label="val_loss", c="green")
    plt.title("epoch-loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.savefig(f"{name}-epoch-accuracy-loss.png")


if __name__ == '__main__':
    # optimizer=sgd data_augmentation=False,cos=False,delay=False  0,0,0
    #print("optimizer=sgd data_augmentation=False,cos=False,delay=False")
    # train(data_augmentation=False,optimizer_name="sgd",cos=False)
    # optimizer=adam,data_augmentation=True,cos=True  1,1,1
    #print("optimizer=adam,data_augmentation=True,cos=True")
    # train(data_augmentation=True, optimizer_name="adam", cos=True)
    # optimizer=adam data_augmentation=False,cos=False  1,0,0
    #print("# optimizer=adam data_augmentation=False,cos=False")
    #train(data_augmentation=False,optimizer_name="adam",cos=False)
    # optimizer=sgd,data_augmentation=True,cos=True,delay=False   0,1,1
    #print("optimizer=sgd,data_augmentation=True,cos=True,delay=False")
    #train(data_augmentation=True,optimizer_name="sgd",cos=True)
    # optimizer=sgd,data_augmentation=False,cos=True,delay=False   0,0,1
    #print("optimizer=sgd,data_augmentation=False,cos=True,delay=False")
    #train(data_augmentation=False,optimizer_name="sgd",cos=True)
    # optimizer=adam,data_augmentation=False,cos=True  1,0,1
    #print("optimizer=adam,data_augmentation=False,cos=True")
    #train(data_augmentation=False,optimizer_name="adam",cos=True)
    # optimizer=adam,data_augmentation=True,cos=False  1,1,0
    #print("optimizer=adam,data_augmentation=True,cos=False")
    #train(data_augmentation=True, optimizer_name="adam", cos=False)
    # optimizer=sgd,data_augmentation=True,cos=False,delay=False   0,1,0
    #print("optimizer=sgd,data_augmentation=True,cos=False,delay=False")
    #train(data_augmentation=True,optimizer_name="sgd",cos=False)

    # optimizer=AdamSGD data_augmentation=True,cos=True,delay=True
    #print("optimizer=AdamSGD data_augmentation=True,cos=True,delay=True")
    #train(data_augmentation=AdamSGD, optimizer_name="AdamSGD", cos=True,weight_decay=True)
    # optimizer=CutMix,data_augmentation=True,cos=True,delay=True
    #print("optimizer=Cutmix,data_augmentation=True,cos=True,delay=True")
    #train(data_augmentation=Cutmix, optimizer_name="Adam", cos=True,weight_decay=True)
    # optimizer=sgd,data_augmentation=False,cos=True,delay=False
    #print("optimizer=sgd,data_augmentation=False,cos=True,delay=False")
    #train(data_augmentation=False, optimizer_name="sgd", cos=True,weight_decay=True)
    # optimizer=sgd,data_augmentation=True,cos=True,delay=False
    print("optimizer=Adam,data_augmentation=True,cos=True,delay=False")
    train(data_augmentation=True, optimizer_name="Adam", cos=True,weight_decay=True)






