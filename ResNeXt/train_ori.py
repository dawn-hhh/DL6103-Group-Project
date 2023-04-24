"""
训练(CPU)
"""

import os
import sys
import json
import time
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from tqdm import tqdm   # 显示进度条模块
from cutmix.cutmix import CutMix
from cutmix.utils import CutMixCrossEntropyLoss
from Cutout import Cutout
from model import resnext50_32x4d, resnet50
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ['CUBLAS_WORKSPACE_CONFIG']=':16:8'
batch_size = 128

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"use device is {device}")

    # data_transform = {
    #     "train": transforms.Compose([
    #                                 transforms.RandomResizedCrop(224),  # 随机裁剪, 再缩放为 224*224
    #                                 transforms.RandomHorizontalFlip(),  # 水平随机翻转
    #                                 transforms.ToTensor(),
    #                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    #     ]),
    #     "val": transforms.Compose([
    #                                 # transforms.Resize((224, 224)),  # 元组(224, 224)
    #                                 transforms.Resize(256),
    #                                 transforms.CenterCrop(224),
    #                                 transforms.ToTensor(),
    #                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    #     ])
    # }
    # data_root = os.path.abspath(os.path.join(os.getcwd(), "../..")) # 读取数据路径
    data_root = os.path.abspath(os.path.join(os.getcwd(), "./"))
    image_path = os.path.join(data_root, "data_set", "snake_data")
    #data preprocessing:

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)
    torch.use_deterministic_algorithms(True)

    transform = transforms.Compose([
        torchvision.transforms.RandomCrop(size=32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.5070, 0.4865, 0.4409],
            std=[0.2673, 0.2564, 0.2761]
        )
    ])

    #transform.transforms.append(Cutout(n_holes=1, length=5))
    
    
    cifar100_train = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    #cifar100_train = CutMix(cifar100_train, num_class=100, beta=1.0, prob=0.5, num_mix=2)


    indices = np.arange(len(cifar100_train))
    train_indices = indices[:40000]
    valid_indices = indices[40000:]
    np.random.shuffle(train_indices)
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
    valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_indices)

    train_loader = torch.utils.data.DataLoader(cifar100_train, batch_size=batch_size, sampler=train_sampler)
    valid_loader = torch.utils.data.DataLoader(cifar100_train, batch_size=batch_size, sampler=valid_sampler)

    cifar100_test = torchvision.datasets.CIFAR100(root='./data', train=False, download=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(cifar100_test, batch_size=batch_size)
    # print(image_path)
    # image_path = data_root + "/data_set/flower_data/"
    # assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    # train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
    #                                      transform=data_transform["train"]
    #                                      )
    train_num = 40000
    val_num = 10000
    # snake_list = train_dataset.class_to_idx
    # class_dict = dict((val, key) for key, val in snake_list.items())
    # json_str = json.dumps(class_dict, indent=4)
    # with open("snake_species.json", 'w') as json_file:
    #     json_file.write(json_str)


    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # 线程数计算
    nw = 0
    print(f"Using {nw} dataloader workers every process.")

    # train_loader = torch.utils.data.DataLoader(train_dataset,
    #                                            batch_size=batch_size,
    #                                            shuffle=True,
    #                                            num_workers=nw
    #                                            )
    # val_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
    #                                    transform=data_transform["val"]
    #                                    )
    # val_num = len(val_dataset)
    # val_loader = torch.utils.data.DataLoader(val_dataset,
    #                                          batch_size=4,
    #                                          shuffle=False,
    #                                          num_workers=nw
    #                                          )
    print(f"Using {train_num} images for training, {val_num} images for validation.")

    # test_data_iter = iter(val_loader)
    # test_image, test_label = next(test_data_iter)

    """ 测试数据集图片"""
    # def imshow(img):
    #     img = img / 2 + 0.5
    #     np_img = img.numpy()
    #     plt.imshow(np.transpose(np_img, (1, 2, 0)))
    #     plt.show()
    # print(' '.join('%5s' % cla_dict[test_label[j].item()] for j in range(4)))
    # imshow(utils.make_grid(test_image))

    net = resnet50()  # 实例化网络
    # load pretrain weights
    # download url: https://download.pytorch.org/models/resnet34-333f7ec4.pth
    # model_weight_path = "./resnext50_32x4d-7cdf4587.pth"
    # assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    # net.load_state_dict(torch.load(model_weight_path, map_location='cpu'))
    # for param in net.parameters():
    #     param.requires_grad = False

    # change fc layer structure
    in_channel = net.fc.in_features
    net.fc = nn.Linear(in_channel, 100)   # (9分类)
    net.to(device)
    # net.to("cpu")   # 直接指定 cpu
    loss_function = nn.CrossEntropyLoss()   # 交叉熵损失
    #loss_function = CutMixCrossEntropyLoss(True)
    
    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    
    optimizer = optim.SGD(params, lr=0.01)
    #optimizer = optim.Adam(params, lr=0.01)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 100)
    #scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.5)

    epochs = 100     # 训练轮数
    save_path = "./myresnext50_3.pth"
    best_accuracy = 0.0
    train_steps = len(train_loader)
    val_steps = len(valid_loader)
    train_acc_set = []
    train_loss_set = []
    val_acc_set = []
    val_loss_set = []

    for epoch in range(epochs):
        net.train()     # 开启Dropout
        running_loss = 0.0
        acc_train = 0.0
        train_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)     # 设置进度条图标
        for step, data in enumerate(train_bar):     # 遍历训练集,
            images, labels = data   # 获取训练集图像和标签
            optimizer.zero_grad()   # 清除历史梯度
            output = net(images.to(device))
            predict_x = torch.max(output, dim=1)[1]
            loss = loss_function(output, labels.to(device))   # 计算损失值

            acc_train += torch.eq(predict_x, labels.to(device)).sum().item()

            loss.backward()     # 方向传播
            optimizer.step()    # 更新优化器参数
            running_loss += loss.item()
            train_bar.desc = "train epoch [{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                      epochs,
                                                                      loss
                                                                      )

        train_accuracy = acc_train / train_num
        train_acc_set.append(train_accuracy)
        train_loss = running_loss / train_steps
        train_loss_set.append(train_loss)
        # 验证
        net.eval()
        acc = 0.0
        one_val_loss = 0.0
        with torch.no_grad():
            val_bar = tqdm(valid_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
                loss = loss_function(outputs, val_labels.to(device))
                one_val_loss += loss.item()
        val_accuracy = acc / val_num
        val_acc_set.append(val_accuracy)
        val_loss = one_val_loss / val_steps
        val_loss_set.append(val_loss)

        scheduler.step()
        print("[epoch %d ] train_loss: %3f  train_accurancy: %3f  val_accurancy: %3f   val_loss: %3f" %
              (epoch + 1, train_loss, train_accuracy, val_accuracy, val_loss))
        if val_accuracy > best_accuracy:  # 保存准确率最高的
            best_accuracy = val_accuracy
            torch.save(net.state_dict(), save_path)

    acc_path = './acc_step200.png'
    save_acc_plot(train_acc_set, val_acc_set, acc_path)
    loss_path = './loss_step200.png'
    save_loss_plot(train_loss_set, val_loss_set, loss_path)

    print("Finished Training.")


def save_loss_plot(train_loss, val_loss, outpath):
    plt.figure(figsize=(16, 10))
    plt.plot(train_loss, color='yellow', label='train loss')
    plt.plot(val_loss, color='red', label='validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(outpath)
    plt.show()


def save_acc_plot(train_loss, val_loss, outpath):
    plt.figure(figsize=(16, 10))
    plt.plot(train_loss, color='blue', label='train accuracy')
    plt.plot(val_loss, color='green', label='validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(outpath)
    plt.show()

if __name__ == '__main__':
    main()

