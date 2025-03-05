import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# 自定义线性层
class CustomLinear(nn.Module):
    def __init__(self, input_dim, output_dim, bias=1):
        super().__init__()
        self.bias = bias
        self.weight = nn.Parameter(torch.Tensor(output_dim, input_dim + self.bias))  # 权重矩阵包含偏置项
        nn.init.xavier_uniform_(self.weight)  # 初始化权重

    def forward(self, x):
        # 输入 x 的形状: (batch_size, input_dim)
        # 添加全 1 列（偏置项）
        x_with_bias = torch.cat([torch.ones(x.size(0),self.bias, device=x.device), x], dim=1)
        # 计算输出: (batch_size, output_dim)
        return torch.mm(x_with_bias, self.weight.t())
    


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
        if x.dim() != 2: 
            raise Exception(f'Expected x dimension: 2, but got dimensioin {x.dim()}')
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
    
    # 自定义逆传播过程
    def custom_backward(self, loss):
        # 反向传播计算梯度
        loss.backward()

        # 遍历每一层，调整学习率并更新权重
        for i, layer in enumerate([self.fc1, self.fc2, self.fc3]):
            if i == 0:
                # 第一层的输入是 x
                O_prev = self.history[-1]["x"]  # 获取上一层的输出（包含偏置项）
            elif i == 1:
                # 第二层的输入是 o1
                O_prev = self.history[-1]["o1"]  # 获取上一层的输出（包含偏置项）
            elif i == 2:
                # 第三层的输入是 o2
                O_prev = self.history[-1]["o2"]  # 获取上一层的输出（包含偏置项）

            # 计算 L2 范数
            norm_O_prev = torch.norm(O_prev, p=2)

            # 计算新的学习率
            alpha = self.beta / (norm_O_prev ** 2 + 1e-8)  # 防止除以零

            # 更新权重
            with torch.no_grad():
                for param in layer.parameters():
                    if param.grad is not None:
                        param -= alpha * param.grad  # 使用调整后的学习率更新权重
                        param.grad.zero_()  # 清空梯度
    
    #用于初始化神经网络权重和偏置
    def reset_parameters(self):
        # 手动设置fc1的权重和偏置
        nn.init.xavier_uniform_(self.fc1.weight)  # Xavier初始化
        nn.init.zeros_(self.fc1.bias)  # 将偏置初始化为0

        nn.init.xavier_uniform_(self.fc2.weight)  # Xavier初始化
        nn.init.zeros_(self.fc2.bias)  # 将偏置初始化为0

        nn.init.xavier_uniform_(self.fc3.weight)  # Xavier初始化
        nn.init.zeros_(self.fc3.bias)  # 将偏置初始化为0

        nn.init.xavier_uniform_(self.output_layer.weight)  # Xavier初始化

    # 收集全部参数 - 用于观察而不能够修改
    def collect_paras(self):
        # 收集当前模型的参数，并将其作为元组的列表添加到历史记录中
        params = [
            (self.fc1.weight.detach().clone(), self.fc1.bias.detach().clone()),
            (self.fc2.weight.detach().clone(), self.fc2.bias.detach().clone()),
            (self.fc3.weight.detach().clone(), self.fc3.bias.detach().clone()),
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
            self.show_element_activation(val_set) 

        return loss
    
    """我们将来所有的补偿都会在这里加入！"""
    # 单次训练函数，要求输入：标准格式训练集，取数据范围，评价函数，优化器，将会返回一个损失
    def single_train_iter(self, data_set, batch_range, criterion, optimizer, use_alfa_normlize):
        self.train()  # 设置模型为训练模式

        inputs = data_set[0][batch_range]
        targets = data_set[1][batch_range]
        # 前向传播
        outputs = self.forward(inputs)
        loss = criterion(outputs, targets)
        # 反向传播计算梯度
        loss.backward()
        # 使用特殊方法
        if use_alfa_normlize:
            for param in self.parameters():
                if param.grad is not None:
                    norm_coef = self.history
                    param.grad *= 1  # 测试修改梯度

        # 使用优化器进行优化
        optimizer.step()
        # 重置梯度
        optimizer.zero_grad()
        
        return loss.detach()


# 训练循环，需要参数：模型、数据集、验证集、评估函数、优化器、学习率更新器、迭代次数、batch_size, 输出循环书数
# 将会返回：模型，验证损失数组、error矩阵
def train_normalnet(model, data_set, val_set, criterion, optimizer, scheduler=None, use_alfa_normalize=False, batch_size=100, num_epochs=1000, show_epoch=20):
    
    model.reset_parameters()

    losses = []
    val_losses = []
    errors = []

    for epoch in range(num_epochs):

        model.train()  # 设置模型为训练模式
        num_datas = len(data_set[0])

        # 在一个batch上进行训练
        batch_range = torch.arange(epoch*batch_size, (epoch+1)*batch_size) % num_datas
        running_loss = model.single_train_iter(data_set, batch_range, criterion, optimizer, use_alfa_normalize)

        if scheduler:
            scheduler.step()  # 更新学习率

        # 获取沿着x，随着迭代次数的error分布
        model.eval()
        outputs_on_val = model(val_set[0]).detach()
        error = torch.sqrt(torch.abs(outputs_on_val - val_set[1]))
        errors.append(error)
        
        if (epoch + 1) % (num_epochs//show_epoch) == 0:  # 如果是输出循环，则输出信息并保留训练信息
            losses.append(running_loss.numpy())
            val_loss = model.validation(val_set, criterion, False).detach()
            val_losses.append(val_loss.numpy())
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss}, Val_Loss: {val_loss}')
            
        model.initialize_history()  # 清空历史
    
    model.eval()

    # 绘制训练损失曲线，并添加标签
    plt.plot(losses, label='Training Loss')
    # 绘制验证损失曲线，并添加标签
    plt.plot(val_losses, label='Validation Loss')
    # 设置图的标题
    plt.title('Training and Validation Loss')
    # 设置 x 轴标签
    plt.xlabel('Epoch')
    # 设置 y 轴标签
    plt.ylabel('Loss')
    # 显示图例
    plt.legend()
    # 显示图形
    plt.show()
    
    # 制作损失矩阵
    errors = torch.stack(errors, dim=1)
    print('Finished Training')
    return model, val_losses, errors.squeeze()