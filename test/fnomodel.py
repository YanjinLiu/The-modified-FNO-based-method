import torch.nn.functional as F
import matplotlib.pyplot as plt
from timeit import default_timer
from utilities3 import *
from Adam import Adam
from scipy import fft

class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()
        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
                                     dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
                                     dtype=torch.cfloat))
        self.weights3 = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
                                     dtype=torch.cfloat))
        self.weights4 = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
                                     dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul3d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1) // 2 + 1,
                             dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)

        # Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x


class FNO3d(nn.Module):
    def __init__(self, modes1, modes2, modes3, width):
        super(FNO3d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.padding = 10  # pad the domain if input is non-periodic
        #self.fc0 = nn.Linear(5, self.width)  # input channel is 3: (a(x, y), x, y)
        self.fc0 = nn.Conv3d(9,self.width,1)

        self.conv0 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv1 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv2 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv3 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.w0 = nn.Conv3d(self.width, self.width, 1)
        self.w1 = nn.Conv3d(self.width, self.width, 1)
        self.w2 = nn.Conv3d(self.width, self.width, 1)
        self.w3 = nn.Conv3d(self.width, self.width, 1)

        #self.fc1 = nn.Linear(self.width, 128)
        #self.fc2 = nn.Linear(128, 4)
        self.fc1 = nn.Conv3d(self.width, 128,1)
        self.fc2 = nn.Conv3d(128, 1, 1)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = x.permute(0, 4, 1, 2, 3)
        pad_z=fft.next_fast_len(x.shape[-1], real=True)-x.shape[-1]
        pad_x = fft.next_fast_len(x.shape[-2], real=True)-x.shape[-2]
        pad_y = fft.next_fast_len(x.shape[-3], real=True)-x.shape[-3]
        # pad_z = judge(x.shape[-1])
        # pad_x = judge(x.shape[-2])
        # pad_y = judge(x.shape[-3])
        x = F.pad(x, [0, pad_z, 0, pad_x, 0, pad_y])
        x = self.fc0(x)
        #x = F.pad(x, [0, 5, 0, self.padding, 0, self.padding])

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        #x = x[..., :-self.padding, :-self.padding, :-5]
        #x = x.permute(0, 2, 3, 4, 1)  # pad the domain if input is non-periodic
        x = x[..., 0:x.shape[-3] - pad_y, 0:x.shape[-2] - pad_x, 0:x.shape[-1] - pad_z]
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x.permute(0,2,3,4,1)

    def get_grid(self, shape, device):
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        return torch.cat((gridx, gridy, gridz), dim=-1).to(device)

def judge(x):
    x1=x/10
    x2=str(round(x1,1))
    x3 = x2.split('.')  # 按“.”将字符串n1进行切片并保存为列表n2
    if x3[1] == '0':  # 判断列表n2的第2个元素是否为0
        pad = 0
        return pad  # 若为0，即为整数，输出整数格式
    else:
        pad = 10-int(x3[1])
        return pad  # 否则输出原数字

class FNO3d_up(nn.Module):
    def __init__(self, modes1, modes2, modes3, width):
        super(FNO3d_up, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.padding = 10  # pad the domain if input is non-periodic
        #self.fc0 = nn.Linear(5, self.width)  # input channel is 3: (a(x, y), x, y)
        self.fc0 = nn.Conv3d(8,self.width,1)

        self.conv0 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv1 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv2 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv3 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.w0 = nn.Conv3d(self.width, self.width, 1)
        self.w1 = nn.Conv3d(self.width, self.width, 1)
        self.w2 = nn.Conv3d(self.width, self.width, 1)
        self.w3 = nn.Conv3d(self.width, self.width, 1)


        #self.fc1 = nn.Linear(self.width, 128)
        #self.fc2 = nn.Linear(128, 4)
        self.fc1 = nn.Conv3d(self.width, 128,1)
        self.fc2 = nn.Conv3d(128, 4, 1)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = x.permute(0, 4, 1, 2, 3)
        pad_z = fft.next_fast_len(x.shape[-1], real=True) - x.shape[-1]
        pad_x = fft.next_fast_len(x.shape[-2], real=True) - x.shape[-2]
        pad_y = fft.next_fast_len(x.shape[-3], real=True) - x.shape[-3]
        # pad_z = judge(x.shape[-1])
        # pad_x = judge(x.shape[-2])
        # pad_y = judge(x.shape[-3])
        x = F.pad(x, [0, pad_z, 0, pad_x, 0, pad_y])
        x = self.fc0(x)
        #x = F.pad(x, [0, 5, 0, self.padding, 0, self.padding])

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2


        #x = x[..., :-self.padding, :-self.padding, :-5]
        #x = x.permute(0, 2, 3, 4, 1)  # pad the domain if input is non-periodic
        x = x[..., 0:x.shape[-3] - pad_y, 0:x.shape[-2] - pad_x, 0:x.shape[-1] - pad_z]
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x.permute(0,2,3,4,1)

    def get_grid(self, shape, device):
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        return torch.cat((gridx, gridy, gridz), dim=-1).to(device)