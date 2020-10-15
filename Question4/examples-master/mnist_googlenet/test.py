import torch
import torch.nn as nn
import torch.optim as optim
import time
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class GlobalAvgPool2d(nn.Module):
#全局平均池化层可通过将池化窗口形状设置成输入的高和宽实现
    def init(self):
        super(GlobalAvgPool2d, self).init()
    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:])
class FlattenLayer(torch.nn.Module):
    def init(self):
        super(FlattenLayer, self).init()
    def forward(self, x): # x shape: (batch, *, *, …)
        return x.view(x.shape[0], -1)
#计算准确率
def evaluate_accuracy(data_iter,net,device = torch.device('cpu')):
#创建 正确率 和 总个数
    acc_sum ,n = torch.tensor([0],dtype=torch.float32,device=device),0
    for X,y in data_iter:
# 适配 设备
        X,y = X.to(device),y.to(device)
# 设置 验证模式
    net.eval()
    with torch.no_grad(): #隔离开 不要计算在计算图内
        y = y.long()#在这里将y转成long确实是不必要的。但是在计算交叉熵时，Pytorch强制要求y是long
        acc_sum += torch.sum((torch.argmax(net(X),dim=1) == y)) # 累计预测正确的个数
        n += y.shape[0] # 累计总的标签个数
    return acc_sum.item() / n
#下载数据 组装好训练数据 测试数据
def load_data_fashion_mnist(batch_size,resize = None,root = './dataset/input/FashionMNIST2065'):
    trans = []
    if resize:

        trans.append(torchvision.transforms.Resize(size=resize))
#将 图片 类型 转化为Tensor类型
        trans.append(torchvision.transforms.ToTensor())
#将图片 增强方式 添加到Compose 类中处理
        transform = torchvision.transforms.Compose(trans)
#读取训练数据
        mnist_train = torchvision.datasets.FashionMNIST(root=root,train=True,download=False,transform = transform)
#读取 测试数据
    mnist_test = torchvision.datasets.FashionMNIST(root = root,train=False,download=False,transform = transform)
#数据加载器 在训练 测试阶段 使用多线程按批采样数据 默认不使用多线程 num_worker 表示设置的线程数量
    train_iter = torch.utils.data.DataLoader(mnist_train,batch_size = batch_size,shuffle = True,num_workers = 2)
    test_iter = torch.utils.data.DataLoader(mnist_test,batch_size = batch_size,shuffle = False,num_workers = 2)
    return train_iter,test_iter

batch_size = 16
#如出现'out of memory'的报错信息，可减小batch_size或resize
train_iter,test_iter = load_data_fashion_mnist(batch_size,224)
def train_fit(net,train_iter,test_iter,batch_size,optimizer,device,num_epochs):
#将读取的数据 拷贝到 指定的GPU上
net = net.to(device)
#设置 损失函数 交叉熵损失函数
loss = torch.nn.CrossEntropyLoss()
#设置训练次数
for epoch in range(num_epochs):
train_l_sum,train_acc_sum,n,batch_count,start = 0.0,0.0,0,0,time.time()
# 读取批量数据 进行训练
for X,y in train_iter:
X = X.to(device)
y = y.to(device)
# 训练结果
y_hat = net(X)
# 计算 预测与标签分布 差异
l = loss(y_hat,y)
# 优化函数 梯度置为零
# 1、因为梯度可以累加
# 2、每批采样的梯度不同，只需记录本次样本的梯度
optimizer.zero_grad()
# 反向求导
l.backward()
# 更新权重参数
optimizer.step()
train_l_sum += l.cpu().item()
#train_acc_sum += (torch.argmax(y_hat,dim = 1) == y).cpu().item()
#将张量元素值累计
train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
n += y.shape[0]
batch_count += 1
test_acc = evaluate_accuracy(test_iter,net)
print(‘epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec’
% (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))
#创建Inception block
class Inception(nn.Module):
def init(self,in_c,c1,c2,c3,c4):
super(Inception,self).init()
self.p1_1 = nn.Conv2d(in_c,c1,kernel_size=1)
self.p2_1 = nn.Conv2d(in_c,c1,kernel_size=1)
self.p2_2 = nn.Conv2d(c2[0],c2[1],kernel_size=3,padding=1)
self.p3_1 = nn.Conv2d(in_c,c3[0],kernel_size=1)
self.p3_2 = nn.Conv2d(c3[0],c3[1],kernel_size=5,padding=2)
self.p4_1 = nn.MaxPool2d(kernel_size=3,stride=1,padding=1)
self.p4_2 = nn.Conv2d(in_c,c4,kernel_size=1)
def forward(self, x):
p1 = F.relu(self.p1_1(x))
p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
p4 = F.relu(self.p4_2(F.relu(self.p4_1(x))))
return torch.cat((p1,p2,p3,p4),dim=1)

b1 = nn.Sequential(
nn.Conv2d(1,64,kernel_size=7,stride=2,padding=3),
nn.ReLU(),
nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
)
b2 = nn.Sequential(
nn.Conv2d(64,64,kernel_size=1),
nn.Conv2d(64,192,kernel_size=3,padding=1),
nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
)
b3 = nn.Sequential(
Inception(192,64,(96,128),(16,32),32),
Inception(256,128,(128,192),(32,96),64),
nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
)
b4 = nn.Sequential(
Inception(192,64,(96,128),(16,32),32),
Inception(512,160,(112,224),(24,64),64),
Inception(512,128,(128,256),(24,64),64),
Inception(512,112,(144,288),(32,64),64),
Inception(528,256,(160,320),(32,128),128),
nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
)
b5 = nn.Sequential(
Inception(832,256,(160,320),(32,128),128),
Inception(832,384,(192,384),(48,128),128),
GlobalAvgPool2d()
)
net = nn.Sequential(b1,b2,b3,b4,b5,
FlattenLayer(),
nn.Linear(1024,10)
)
X = torch.rand(1,1,96,96)
for blk in net.children():
X = blk(X)
lr,num_epochs = 0.001,5
optimizer = torch.optim.Adam(net.parameters(),lr= lr)
train_fit(net,train_iter,test_iter,batch_size,optimizer,device=device,num_epochs = num_epochs)