import torchvision #导入pytorch里的视觉库
import torch
from torch import nn
#from torch.utils.tensorboard import SummaryWriter #导入网络的训练记录，记录loss变化

#加载自定义网络模型
from ultralytics import YOLO

#加载数据集处理函数
from torch.utils.data import DataLoader

#加载数据集
train_set=torchvision.datasets.CIFAR10(root="./CIRAR10_set",train=True,transform=torchvision.transforms.ToTensor(),
                                       download=False)
test_set=torchvision.datasets.CIFAR10(root="./CIRAR10_set",train=False,transform=torchvision.transforms.ToTensor(),
                                      download=False)

#数据集长度
train_set_size=len(train_set)
test_set_size=len(test_set)
#print("训练集的长度： {}".format((train_set_size)))
#print("测试集的长度： {}".format((test_set_size)))

#利用DataLoader加载数据集
train_dataloader=DataLoader(train_set,batch_size=64) #batch_size就是训练时每次输入多少张图片
test_dataloader=DataLoader(test_set,batch_size=64)

#创建新的网络模型
model = YOLO("yolov8-CFF.pt")

#损失函数
loss_fn=nn.CrossEntropyLoss() #交叉熵损失函数，常用于分类问题

#创建优化器
learning_rate=0.01
optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate)

#设置训练网络的参数
total_train_step=0
total_test_step=0
epoch=10 #定义训练循环次数

#添加Tensorboard，记录训练过程中的损失函数变化
#writer=SummaryWriter("logs")
#显示  tensorboard --logdir=logs

for i in range(epoch):
    print("-------第{}轮训练开始-------".format(i+1))

    #训练步骤开始
    model.train()  #当网络中有dropout或者batchnorm层的时候需要写
    for data in train_dataloader:
        imgs,targets=data
        outputs=model(imgs)
        loss=loss_fn(outputs,targets)

        #优化器优化
        optimizer.zero_grad() #清零梯度优化器
        loss.backward() #反向传播，根据loss计算对权重系数的偏导数
        optimizer.step() #更新权重

        total_train_step=total_train_step+1
        if total_train_step % 100 ==0:
            print("训练次数：{}，损失值：{}".format(total_train_step,loss))
            #writer.add_scalar("train_loss",loss.item(),total_train_step)

    #开始测试步骤
    model.eval() #当网络中有dropout或者batchnorm层的时候需要写
    total_test_loss=0
    with torch.no_grad():
        for data in test_dataloader:
            imgs,targets=data
            outputs=model(imgs)
            loss=loss_fn(outputs,targets)

            total_test_loss=total_test_loss+loss

    print("整体测试集上的loss：{}".format(total_test_loss))
    #writer.add_scalar("test_loss",total_test_loss,total_test_step)
    total_test_step=total_test_step+1

    #保存每一轮训练的结果
    torch.save(model,"sci_{}.pth".format(i))
    print("模型已保存")

#writer.close()



