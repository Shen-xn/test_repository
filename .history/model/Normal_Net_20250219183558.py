import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.autograd import Function


class CustomLinearFunction(Function):
    @staticmethod
    def forward(ctx, 
               input,  # O_prev
               weight, bias, 
               prev_layer_info,  # 存储前一层信息的元组 (O_prev_prev, A_prev, activation_fn_prev)
               use_grad_norm=False, use_combination=False, gamma=None):  # 定义这一块的正向传播过程
        """
        输入 input: (batch_size, input_dim + 1)
        权重 weight: (output_dim, input_dim + 1)
         prev_layer_info: 元组包含：
            O_prev_prev_with_bias: 前前层输出（带偏置）
            A_prev: 前一层激活前输出
            activation_fn: 前一层使用的激活函数
        ctx的作用见注释
        """

        # 添加本层偏置
        ctx.bias = int(bias)
        O_k_minus1_with_bias = torch.cat([
            torch.ones(input.size(0), ctx.bias, device=input.device),
            input
        ], dim=1)

        # 保存反向传播需要的中间变量
        ctx.save_for_backward(O_k_minus1_with_bias, weight)
        ctx.prev_layer_info = prev_layer_info  # O_prev_prev_with_bias, A_prev, activation_fn_prev
        ctx.use_grad_norm = use_grad_norm
        ctx.use_combination = use_combination
        ctx.gamma = gamma

        # 返回输出
        return torch.mm(O_k_minus1_with_bias, weight.t())

    @staticmethod
    def backward(ctx, grad_output):  # 定义逆向传播过程
        # 输入梯度 grad_output: (batch_size, output_dim) - 这个就是我们的delta
        O_k_minus1_with_bias, weight = ctx.saved_tensors  # (batch_size, output_dim), (input_dim + 1, output_dim)
        O_prev_prev, A_prev, activation_fn = ctx.prev_layer_info

         # 计算传播到前层的梯度，这个除非是最后一层不然必须计算
        if ctx.needs_input_grad[0] or ctx.use_combination:
            grad_O_k = torch.mm(grad_output, weight)[:, ctx.bias:]

        # 计算处理补偿所需的可复用材料（如果提供了基本材料）
        if A_prev is not None and activation_fn is not None:
            # 计算激活函数导数
            with torch.enable_grad():
                A_prev = A_prev.detach().requires_grad_()
                activated = activation_fn(A_prev)
                f_prime, = torch.autograd.grad(
                    activated, A_prev,
                    grad_outputs=torch.ones_like(activated),
                    create_graph=torch.is_grad_enabled(),
                    retain_graph=True
                )
            # 应用激活函数导数
            delta_k_minus1 = grad_O_k * f_prime
        else:  # 如果没有则不需要计算，则是None
            f_prime = None
            delta_k_minus1 = None
        
        # 计算两种算法都需要的必要材料
        if ctx.use_grad_norm or ctx.use_combination:
            norms_O_prev = torch.norm(O_k_minus1_with_bias, dim=1, keepdim=True).square()

        # 应用梯度归一化
        if ctx.use_grad_norm:
            # 计算学习率归一化系数
            alphas = (1 / (norms_O_prev)).view(-1, 1)       # (batch_size, 1)
            grad_output /= alphas

        # 应用梯度补偿
        # if ctx.use_combination:
        if ctx.use_combination and torch.randint(0, 2, (1,)) < 1:
            if O_prev_prev is not None and f_prime is not None and delta_k_minus1 is not None:
                # 计算范数比
                norms_O_prev_prev = torch.norm(O_prev_prev, dim=1, keepdim=True).square() + 1
                ratio = (norms_O_prev_prev / (norms_O_prev))

                # 构造带偏置的delta
                delta_O_temp = torch.cat([
                    torch.zeros(grad_output.size(0), ctx.bias, device=grad_output.device),
                    delta_k_minus1 * f_prime
                ], dim=1)

                # 计算补偿项
                compensation = ctx.gamma * ratio * torch.mm(delta_O_temp, weight.t())

                # 应用补偿
                grad_output -= compensation
            # 如果要求使用补偿却没有给O_prev_prev_with_bias就会报错
            else:
                raise Exception(f'Wrrong situation when use_combination = {ctx.use_combination} but O_prev_prev_with_bias = {O_prev_prev}, f_prime = {f_prime} and delta_k_minus1 = {delta_k_minus1}')
        
        # 计算权重梯度
        if ctx.needs_input_grad[1]:
            grad_weight = torch.mm(grad_output.t(), O_k_minus1_with_bias)

        # 返回梯度（包含前层需要的delta）
        return (
            grad_O_k if ctx.needs_input_grad[0] else None,
            grad_weight,
            None,
            None,  # prev_layer_info不需要梯度
            None, None, None  # 超参数不需要梯度
        )


# 自定义线性层
class CustomLinear(nn.Module):
    def __init__(self, input_dim, output_dim, bias=1, use_grad_norm=False, use_combination=False, gamma=1):
        super().__init__()
        # 层超参数
        self.use_grad_norm = use_grad_norm
        self.use_combination = use_combination
        self.gamma = gamma
        self.bias = bias

        # 创建权重矩阵并初始化(output_dim, input_dim + 1)
        self.weight = nn.Parameter(torch.Tensor(output_dim, input_dim + self.bias))  # 权重矩阵包含偏置项
        nn.init.normal_(self.weight, mean=0, std=0.1)  # 初始化权重

    def forward(self, o_k, prev_layer_info=(None, None, None)):
        # 保证输入 x 的形状: (batch_size, input_dim)
        x_dim = o_k.dim()
        if x_dim != 2:
            if x_dim == 1:
                o_k.view(1, -1)
            else:
                raise Exception(f'Expected x dimension: 2 or 1, but got dimensioin {o_k.dim()}')
        # 计算输出: (batch_size, output_dim) = (batch_size, input_dim + 1)*(output_dim, input_dim + 1).T, 同时定义这个过程的逆传播过程，见CustomLinearFunction类，
        # 输入向量x中的1我们会在function内部添加
        return CustomLinearFunction.apply(o_k, self.weight, self.bias, prev_layer_info, self.use_grad_norm, self.use_combination, self.gamma)
    


# 模型基本结构
class NormNet(nn.Module):
    # 初始化神经网络类, 需要参数(激活函数，隐藏层维度，是否使用batchnorm)
    def __init__(self, activation_func, layer_dim, usebatchnorm=False, use_grad_norm=False, use_combination=False, gamma=0):  
        
        super(NormNet, self).__init__()  # 继承父类的初始化
        
        # 初始化网络层
        self.__dim = layer_dim

        self.fc1 = CustomLinear(1, self.__dim, bias=1, use_grad_norm=use_grad_norm, use_combination=False, gamma=gamma)  # 输入层到隐层的线性变换 (batch_size, 1)->(batch_size, dim)
        self.fc2 = CustomLinear(self.__dim, self.__dim, bias=1, use_grad_norm=use_grad_norm, use_combination=use_combination, gamma=gamma) 
        self.fc3 = CustomLinear(self.__dim, self.__dim, bias=1, use_grad_norm=use_grad_norm, use_combination=use_combination, gamma=gamma)
        self.fc4 = CustomLinear(self.__dim, 1, bias=1, use_grad_norm=use_grad_norm, use_combination=use_combination, gamma=gamma)
        # self.output_layer = CustomLinear(self.__dim, 1, bias=1, use_grad_norm=False, use_combination=use_combination)  # (batch_size, dim)->(batch_size, 1)

        self.batchnorm1 = nn.BatchNorm1d(
            num_features=self.__dim,
            eps=1e-10,  # 为避免除以零的一个小值
            momentum=0.5,  # 运行时统计量的动量
            affine=True,  # 是否使用可学习的仿射参数（gamma 和 beta）
            track_running_stats=True  # 是否跟踪运行时统计量
        )
        self.batchnorm2 = nn.BatchNorm1d(
            num_features=self.__dim,
            eps=1e-10,
            momentum=0.5,
            affine=True,
            track_running_stats=True
        )
          # 添加 BatchNorm1d 层

        self.batchnorm3 = nn.BatchNorm1d(
            num_features=self.__dim,
            eps=1e-10,
            momentum=0.5,
            affine=True,
            track_running_stats=True
        )

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
        a2 = self.fc2(o1, prev_layer_info=(x, a1, self.activation_func)) 
        o2 = self.activation_func(a2)
        if self.usebatchnorm == True:
            o2 = self.batchnorm2(o2)  # 应用 BatchNorm1d 层
        a3 = self.fc3(o2, prev_layer_info=(o1, a2, self.activation_func))
        o3 = self.activation_func(a3)
        if self.usebatchnorm == True:
            o3 = self.batchnorm3(o3)  # 应用 BatchNorm1d 层
        a4 = self.fc4(o3, prev_layer_info=(o2, a3, self.activation_func))

        self.history.append({"x": x.detach(), "a1": a1.detach(), "o1": o1.detach(),   # 一旦改变深度，这里需要修改！
                             "a2": a2.detach(), "o2": o2.detach(), "a3": a3.detach(), "o3": o3.detach()})  # 用于收集输出历史 [{"x":(batch_size, 1),...,"v":(batch_size, 1)}, {}, {}, ...]
        return a4


    #用于初始化神经网络权重和偏置
    def reset_parameters(self):
        # 手动设置fc1的权重和偏置
        nn.init.normal_(self.fc1.weight, mean=0, std=0.1)  # 初始化
        nn.init.normal_(self.fc2.weight, mean=0, std=0.1)
        nn.init.normal_(self.fc3.weight, mean=0, std=0.1)
        nn.init.normal_(self.fc4.weight, mean=0, std=0.1)
        # nn.init.normal_(self.output_layer.weight, mean=0, std=0.1)  # Xavier初始化

    # 收集全部参数 - 用于观察而不能够修改
    def collect_paras(self):
        # 收集当前模型的参数，并将其作为元组的列表添加到历史记录中
        params = [
            (self.fc1.weight.detach().clone(), self.fc1.bias.detach().clone()),
            (self.fc2.weight.detach().clone(), self.fc2.bias.detach().clone()),
            (self.fc3.weight.detach().clone(), self.fc3.bias.detach().clone()),
            (self.fc4.weight.detach().clone(), self.fc3.bias.detach().clone())
            # (self.output_layer.weight.detach().clone())
        ]
        self.param_history.append(params)

    # 初始化历史
    def initialize_history(self):
        self.history = []

    # 用于显示最后一个历史batch中所有的某个神经元上的的输出随输入x的变化，注意如果最后一次输入了一个单个数据，则会得到单个点！
    @staticmethod
    def show_activation(vector, name, x_val):
        for i in vector:
            j = 0
            plt.plot(x_val, i, label = (f"element{j}"))
            plt.title(f"Activation of {name}")
            plt.xlabel("X values")
            plt.ylabel("Out put")
            j += 1
        plt.show()

    def show_element_activation(self, val_set):
        history = self.history
        o1 = history[0]["o1"].T
        o2 = history[0]["o2"].T
        v = history[0]["o3"].T
        x_val = val_set[0]
        self.show_activation(o1, "o1", x_val)
        self.show_activation(o2, "o2", x_val)
        self.show_activation(v, "o3", x_val)

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
    
    # 单次训练函数，要求输入：标准格式训练集，取数据范围，评价函数，优化器，将会返回一个损失
    def single_train_iter(self, data_set, batch_range, criterion, optimizer):
        self.train()  # 设置模型为训练模式

        inputs = data_set[0][batch_range]
        targets = data_set[1][batch_range]
        # 前向传播
        outputs = self.forward(inputs)
        # 计算误差函数
        loss = criterion(outputs, targets)

        # 重置梯度
        for item in optimizer:
            item.zero_grad()
        
        # 反向传播计算梯度
        loss.backward()

        # 使用优化器进行优化
        for item in optimizer:
            item.step()
        
        return loss.detach()


# 训练循环，需要参数：模型、数据集、验证集、评估函数、优化器、学习率更新器、迭代次数、batch_size, 输出循环书数
# 将会返回：模型，验证损失数组、error矩阵
def train_normalnet(model, data_set, val_set, criterion, optimizer, scheduler=None, batch_size=100, num_epochs=1000, show_epoch=20, print_info=True):
    
    model.reset_parameters()
    # 确保opt装在列表里
    if type(optimizer) != list:
        optimizer = [optimizer]
    # 加入初始的loss
    initial_loss = model.validation(val_set, criterion, False).detach().numpy()
    losses = [initial_loss]
    val_losses = [initial_loss]

    errors = []
    images = []
    model.eval()
    # 加入初始的误差和像
    outputs_on_val = model(val_set[0]).detach()
    error = torch.sqrt(torch.abs(outputs_on_val - val_set[1]))
    errors.append(error)
    images.append(outputs_on_val)

    for epoch in range(num_epochs):

        model.train()  # 设置模型为训练模式
        num_datas = len(data_set[0])

        # 在一个batch上进行训练 - 在这里可以按照顺序遍历，也可以使用随机的100个索引
        # batch_range = torch.arange(epoch*batch_size, (epoch+1)*batch_size) % num_datas
        batch_range = torch.randint(0, num_datas, (batch_size,))
        running_loss = model.single_train_iter(data_set, batch_range, criterion, optimizer)

        if scheduler:
            scheduler.step()  # 更新学习率

        # 获取沿着x，随着迭代次数的error分布
        model.eval()
        outputs_on_val = model(val_set[0]).detach()
        error = torch.sqrt(torch.abs(outputs_on_val - val_set[1]))
        errors.append(error)
        
        
        if (epoch + 1) % (num_epochs//show_epoch) == 0:  # 如果是输出循环，则输出信息并保留训练信息
            # 保存训练集上的损失
            losses.append(running_loss.numpy())
            # 保存验证集上的损失
            val_loss = model.validation(val_set, criterion, False).detach()
            val_losses.append(val_loss.numpy())
            # 同时，保变换的像
            images.append(outputs_on_val)
            # 输出一次信息
            if print_info:
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
    images = torch.stack(images, dim=1)
    print('Finished Training')
    return model, val_losses, errors.squeeze(), images.squeeze()

import torch
from torch.optim import Optimizer


class CustomOpt(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(CustomOpt, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # 获取梯度
                grad = p.grad.data

                # 获取状态
                state = self.state[p]

                # 初始化状态
                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p.data)  # 一阶矩
                    state['v'] = torch.zeros_like(p.data)  # 二阶矩

                # 从状态中读取值
                m, v = state['m'], state['v']
                beta1, beta2 = group['betas']

                # 更新步数
                state['step'] += 1

                # 更新一阶矩和二阶矩
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # 偏差修正
                m_hat = m / (1 - beta1 ** state['step'])
                v_hat = v / (1 - beta2 ** state['step'])

                # 更新参数
                p.data.addcdiv_(m_hat, v_hat.sqrt() + group['eps'], value=-group['lr'])

        return loss
    

# 带有梯度截断的随机梯度下降
class CustomSGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0, weight_decay=0, clip_value=None):
        """
        自定义 SGD 优化器，支持动量、权重衰减和梯度裁剪。

        参数:
        - params: 需要优化的参数（通常是 model.parameters()）。
        - lr: 学习率（默认 0.01）。
        - momentum: 动量系数（默认 0）。
        - weight_decay: 权重衰减系数（L2 正则化，默认 0）。
        - clip_value: 梯度裁剪的阈值（默认 None，表示不裁剪）。
        """
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, clip_value=clip_value)
        super(CustomSGD, self).__init__(params, defaults)

    def step(self, closure=None):
        """
        执行一次参数更新。
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            # 获取超参数
            lr = group['lr']
            momentum = group['momentum']
            weight_decay = group['weight_decay']
            clip_value = group['clip_value']

            for p in group['params']:
                if p.grad is None:
                    continue

                # 获取梯度
                grad = p.grad.data

                # 梯度裁剪
                if clip_value is not None:
                    torch.nn.utils.clip_grad_norm_(p, clip_value)

                # 权重衰减（L2 正则化）
                if weight_decay != 0:
                    grad.add_(p.data, alpha=weight_decay)

                # 动量更新
                if momentum != 0:
                    param_state = self.state[p]  # 获取参数的状态
                    if 'momentum_buffer' not in param_state:
                        # 初始化动量缓冲区
                        buf = param_state['momentum_buffer'] = torch.clone(grad).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(grad)  # 更新动量缓冲区
                    grad = buf  # 使用动量更新梯度

                # 更新参数
                p.data.add_(grad, alpha=-lr)

        return loss