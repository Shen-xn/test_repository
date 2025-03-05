import torch.nn as nn
import matplotlib.pyplot as plt


class NormNet(nn.Module):
    # 初始化神经网络类, 需要参数(激活函数，隐藏层维度，是否使用batchnorm)
    def __init__(self, activation_func, layer_dim, usebatchnorm=False):  
        
        super(NormNet, self).__init__()  # 继承父类的初始化
        
        # 初始化网络层
        self.__dim = layer_dim
        self.fc1 = nn.Linear(1, self.__dim, bias=True) # 输入层到隐层的线性变换
        self.fc2 = nn.Linear(self.__dim, self.__dim, bias=True)
        self.fc3 = nn.Linear(self.__dim, self.__dim, bias=True)
        self.output_layer = nn.Linear(self.__dim, 1, bias=False)

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
                             "a2": a2.detach(), "o2": o2.detach(), "v": v.detach()})
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

    def show_paras(self, num_epochs):
        param_norms = []  # 存储每个 epoch 参数范数的列表
        for epoch_params in self.param_history:
            for param_tuple in epoch_params:
                try:
                    weight, bias = param_tuple
                except ValueError:
                    weight = param_tuple[0]
                    
                norm_sum += weight.norm().item() + bias.norm().item()
            param_norms.append(norm_sum)

        plt.plot(range(num_epochs), param_norms, marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Parameter Norm')
        plt.title('Parameter Norm Over Epochs')
        plt.grid(True)
        plt.show()

    def initialize_history(self):
        self.history = []

    def show_history(self, a=-2, b=2):
        for i in self.history:
            # for key in i:
            #     print(f"{key}: {i[key]}")
            plt.plot(i["a1"], color='blue', label='a1')
            plt.plot(i["o1"], color='red', label='o1')
            plt.ylim(a, b)
            plt.show()
            plt.plot(i["o1"], color='green', label='o1')
            plt.plot(i["a2"], color='blue', label='a2')
            plt.plot(i["o2"], color='red', label='o2')
            plt.ylim(a, b)
            plt.show()
            plt.plot(i["o2"], color='green', label='o2')
            plt.plot(i["v"], color='blue', label='v')
            plt.ylim(a, b)
            plt.show()