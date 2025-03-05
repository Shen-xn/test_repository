import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# 模型基本结构
class NormNet(nn.Module):
    # 初始化神经网络类, 需要参数(激活函数，隐藏层维度，是否使用batchnorm)
    def __init__(self, activation_func, layer_dim, usebatchnorm=False):  
        
        super(NormNet, self).__init__()  # 继承父类的初始化
        
        # 初始化网络层
        self.__dim = layer_dim
        self.fc1 = nn.Linear(1, self.__dim, bias=True)  # 输入层到隐层的线性变换 (batch_size, 1)->(batch_size, dim)
        self.fc2 = nn.Linear(self.__dim, self.__dim, bias=True) 
        self.fc3 = nn.Linear(self.__dim, self.__dim, bias=True)
        self.output_layer = nn.Linear(self.__dim, 1, bias=False)  # (batch_size, dim)->(batch_size, 1)

        self.batchnorm1 = nn.BatchNorm1d(
            num_features=self.__dim,
            eps=1e-06,  # 为避免除以零的一个小值
            momentum=0.9,  # 运行时统计量的动量
            affine=True,  # 是否使用可学习的仿射参数（gamma 和 beta）
            track_running_stats=True  # 是否跟踪运行时统计量
        )
        self.batchnorm2 = nn.BatchNorm1d(
            num_features=self.__dim,
            eps=1e-06,
            momentum=0.9,
            affine=True,
            track_running_stats=True
        )
          # 添加 BatchNorm1d 层

        self.usebatchnorm = usebatchnorm
        self.activation_func = activation_func
        self.history = []
        self.param_history = []
        
    # 前向传播过程
    def forward(self, x):
        
        a1 = self.fc1(x)
        o1 = self.activation_func(a1)
        if self.usebatchnorm == True:
            o1 = self.batchnorm1(o1)  # 应用 BatchNorm1d 层
        a2 = self.fc2(o1) 
        o2 = self.activation_func(a2)
        if self.usebatchnorm == True:
            o2 = self.batchnorm2(o2)  # 应用 BatchNorm1d 层
        v = self.fc3(o2)
        o3 = self.output_layer(v)

        self.history.append({"x": x.detach(), "a1": a1.detach(), "o1": o1.detach(), 
                             "a2": a2.detach(), "o2": o2.detach(), "v": v.detach()})  # 用于收集输出历史 [{"x":(batch_size, 1),...,"v":(batch_size, 1)}, {}, {}, ...]
        return o3
    
    #用于初始化神经网络权重和偏置
    def reset_parameters(self):
        # 手动设置fc1的权重和偏置
        nn.init.xavier_uniform_(self.fc1.weight)  # Xavier初始化
        nn.init.zeros_(self.fc1.bias)  # 将偏置初始化为0

        nn.init.xavier_uniform_(self.fc2.weight)  # Xavier初始化
        nn.init.zeros_(self.fc2.bias)  # 将偏置初始化为0

        nn.init.xavier_uniform_(self.fc6.weight)  # Xavier初始化
        nn.init.zeros_(self.fc6.bias)  # 将偏置初始化为0

        nn.init.xavier_uniform_(self.output_layer.weight)  # Xavier初始化

    # 收集全部参数 - 用于观察而不能够修改
    def collect_paras(self):
        # 收集当前模型的参数，并将其作为元组的列表添加到历史记录中
        params = [
            (self.fc1.weight.detach().clone(), self.fc1.bias.detach().clone()),
            (self.fc2.weight.detach().clone(), self.fc2.bias.detach().clone()),
            (self.fc6.weight.detach().clone(), self.fc6.bias.detach().clone()),
            (self.output_layer.weight.detach().clone())
        ]
        self.param_history.append(params)

    # 初始化历史
    def initialize_history(self):
        self.history = []

    # 用于显示最后一个历史batch中所有的某个神经元上的的输出随输入x的变化，注意如果最后一次输入了一个单个数据，则会得到单个点！
    def show_element_activation(self, val_set):
        history = self.history
        o1 = history[0]["o1"].T
        o2 = history[0]["o2"].T
        for i in o1:
            j = 0
            plt.plot(val_set[0], i, label = (f"element{j}"))
            plt.title("Activation of layer 1")
            plt.xlabel("X values")
            plt.ylabel("Out put")
            j += 1
        plt.show()
        for i in o2:
            j = 0
            plt.plot(val_set[0], i, label = (f"element{j}"))
            plt.title("Activation of layer 2")
            plt.xlabel("X values")
            plt.ylabel("Out put")
            j += 1
        plt.show()

    # 用于在验证集上测试模型，并输出loss，要求按照“标准格式”的训练集作为参数
    def validation(self, val_set, criterion, take_history=False):
        self.eval()
        self.initialize_history()
        input = val_set[0]
        output = self.forward(input)
        target = val_set[1]
        loss = criterion(output, target)
            
        if take_history:  # 如果需要输出各个神经元在验证集上的历史输出，则调用show_element_activation方法
            self.show_element_activation(self, val_set)

        return loss
    
    """我们将来所有的补偿都会在这里加入！"""
    # 单次训练函数，要求输入：标准格式训练集，取数据范围，评价函数，优化器，将会返回一个损失
    def single_train_iter(self, data_set, batch_range, criterion, optimizer):
        self.train()  # 设置模型为训练模式

        inputs = data_set[0][batch_range]
        targets = data_set[1][batch_range]
        # 前向传播
        outputs = self.forward(inputs)
        loss = criterion(outputs, targets)
        # 反向传播和优化
        
        loss.backward()
        optimizer.step()
        running_loss += loss.detach()
        optimizer.zero_grad()
        return loss


# 训练循环
def train_normalnet(model, data_set, val_set, criterion, optimizer, scheduler, num_epochs=1000, batch_size=100, show_epoch=20):
    
    model.reset_parameters()

    losses = []
    val_losses = []
    errors = []

    for epoch in range(num_epochs):

        model.train()  # 设置模型为训练模式
        num_datas = len(data_set[0])

        # 在一个batch上进行训练
        batch_range = torch.arange(epoch*batch_size, (epoch+1)*batch_size) % num_datas
        running_loss = model.single_train_iter(data_set, batch_range, criterion, optimizer)

        scheduler.step()  # 更新学习率

        # 获取沿着x，随着迭代次数的error分布
        model.eval()
        outputs_on_val = model(val_set[0]).detach()
        error = torch.sqrt(torch.abs(outputs_on_val - val_set[1]))
        errors.append(error)
        
        if (epoch + 1) % show_epoch == 0:  # 如果是输出循环，则输出信息并保留训练信息
            losses.append(running_loss.numpy())

            val_loss = model.validation(model, val_set, criterion).detach()
            val_losses.append(val_loss.numpy())
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss}, Val_Loss: {val_loss}')
            
        model.initialize_history()
        # model.collect_paras()
    
    model.eval()
    plt.plot(losses)
    plt.plot(val_losses)
    plt.show()
    errors = torch.stack(errors, dim=1)
    
    print('Finished Training')
    return model, val_losses, errors.squeeze()