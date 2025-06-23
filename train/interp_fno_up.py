"""
@author: Zongyi Li and Daniel Zhengyu Huang
"""

import torch.nn.functional as F
import matplotlib.pyplot as plt
from timeit import default_timer
from utilities3 import *
from Adam import Adam
from scipy import fft

torch.manual_seed(0)
np.random.seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True

################################################################
# fourier layer
################################################################
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


################################################################
# configs
################################################################
PATH = "sort_D_a_up/"
TRAIN_PATH1 = PATH+'input_a.mat'
TRAIN_PATH2 = PATH+'input_ux.mat'
TRAIN_PATH3 = PATH+'input_uy.mat'
TRAIN_PATH4 = PATH+'input_uz.mat'
TRAIN_PATH5 = PATH+'input_p.mat'
TRAIN_PATH6 = PATH+'input_ui.mat'
TRAIN_PATH7 = PATH+'input_x.mat'
TRAIN_PATH8 = PATH+'input_y.mat'
TRAIN_PATH9 = PATH+'input_z.mat'

ntotal = 960
ntrain = 582
nvalid = 194
ntest = 184

batch_size = 1
learning_rate = 0.001

epochs = 500
step_size = 50
gamma = 0.5

modes1 = 6
modes2 = 16
modes3 = 8
width = 32

r = 1
sub=1


path = 'test24'
# path = 'ns_fourier_V100_N'+str(ntrain)+'_ep' + str(epochs) + '_m' + str(modes) + '_w' + str(width)
path_model = 'model/'+path
path_train_err = 'results/'+path+'train.txt'
path_test_err = 'results/'+path+'test.txt'
path_image = 'image/'+path
savepath='pred/sort_D_a_up_lr001'

################################################################
# load data
################################################################
reader1 = MatReader(TRAIN_PATH1)
reader2 = MatReader(TRAIN_PATH2)
reader3 = MatReader(TRAIN_PATH3)
reader4 = MatReader(TRAIN_PATH4)
reader5 = MatReader(TRAIN_PATH5)
reader6 = MatReader(TRAIN_PATH6)
reader7 = MatReader(TRAIN_PATH7)
reader8 = MatReader(TRAIN_PATH8)
reader9 = MatReader(TRAIN_PATH9)
#
input_a = reader1.data['input_a']
input_ux = reader2.data['input_ux']
input_uy = reader3.data['input_uy']
input_uz = reader4.data['input_uz']
input_p = reader5.data['input_p']
input_ui = reader6.data['input_ui']
input_x = reader7.data['input_x']
input_y = reader8.data['input_y']
input_z = reader9.data['input_z']

input_a=input_a.reshape((input_a.size))
input_ux=input_ux.reshape((input_ux.size))
input_uy=input_uy.reshape((input_uy.size))
input_uz=input_uz.reshape((input_uz.size))
input_p=input_p.reshape((input_p.size))
input_ui=input_ui.reshape((input_ui.size))
input_x=input_x.reshape((input_x.size))
input_y=input_y.reshape((input_y.size))
input_z=input_z.reshape((input_z.size))




for i in range(input_a.size):
    input_a[i] = torch.tensor(input_a[i][:, ::sub, ::sub, ::sub], dtype=torch.float).permute(3, 2, 1, 0)
    input_ux[i] = torch.tensor(input_ux[i][:, ::sub, ::sub, ::sub], dtype=torch.float).permute(3, 2, 1, 0)
    input_uy[i] = torch.tensor(input_uy[i][:, ::sub, ::sub, ::sub], dtype=torch.float).permute(3, 2, 1, 0)
    input_uz[i] = torch.tensor(input_uz[i][:, ::sub, ::sub, ::sub], dtype=torch.float).permute(3, 2, 1, 0)
    input_p[i] = torch.tensor(input_p[i][:, ::sub, ::sub, ::sub], dtype=torch.float).permute(3, 2, 1, 0)
    input_ui[i] = torch.tensor(input_ui[i][:, ::sub, ::sub, ::sub], dtype=torch.float).permute(3, 2, 1, 0)
    input_x[i] = torch.tensor(input_x[i][:, ::sub, ::sub, ::sub], dtype=torch.float).permute(3, 2, 1, 0)
    input_y[i] = torch.tensor(input_y[i][:, ::sub, ::sub, ::sub], dtype=torch.float).permute(3, 2, 1, 0)
    input_z[i] = torch.tensor(input_z[i][:, ::sub, ::sub, ::sub], dtype=torch.float).permute(3, 2, 1, 0)

train_a = input_a[:ntrain]
train_ux = input_ux[:ntrain]
train_uy = input_uy[:ntrain]
train_uz = input_uz[:ntrain]
train_p = input_p[:ntrain]
train_ui = input_ui[:ntrain]
train_x = input_x[:ntrain]
train_y = input_y[:ntrain]
train_z = input_z[:ntrain]

valid_a = input_a[ntrain:ntrain+nvalid]
valid_ux = input_ux[ntrain:ntrain+nvalid]
valid_uy = input_uy[ntrain:ntrain+nvalid]
valid_uz = input_uz[ntrain:ntrain+nvalid]
valid_p = input_p[ntrain:ntrain+nvalid]
valid_ui = input_ui[ntrain:ntrain+nvalid]
valid_x = input_x[ntrain:ntrain+nvalid]
valid_y = input_y[ntrain:ntrain+nvalid]
valid_z = input_z[ntrain:ntrain+nvalid]

test_a = input_a[ntrain+nvalid:ntotal]
test_ux = input_ux[ntrain+nvalid:ntotal]
test_uy = input_uy[ntrain+nvalid:ntotal]
test_uz = input_uz[ntrain+nvalid:ntotal]
test_p = input_p[ntrain+nvalid:ntotal]
test_ui = input_ui[ntrain+nvalid:ntotal]
test_x = input_x[ntrain+nvalid:ntotal]
test_y = input_y[ntrain+nvalid:ntotal]
test_z = input_z[ntrain+nvalid:ntotal]


def my_collate(batch):
    train_a = [item[0] for item in batch]
    train_ui = [item[1] for item in batch]
    train_x = [item[2] for item in batch]
    train_y = [item[3] for item in batch]
    train_z = [item[4] for item in batch]
    train_ux = [item[5] for item in batch]
    train_uy = [item[6] for item in batch]
    train_uz = [item[7] for item in batch]
    train_p = [item[8] for item in batch]
    return [train_a, train_ui, train_x, train_y, train_z, train_ux,train_uy,train_uz,train_p]

train_loader = torch.utils.data.DataLoader(list(zip(train_a, train_ui, train_x, train_y, train_z,train_ux,train_uy,train_uz,train_p)), batch_size=batch_size,
                                           shuffle=True, collate_fn=my_collate)
valid_loader = torch.utils.data.DataLoader(list(zip(valid_a, valid_ui, valid_x, valid_y, valid_z,valid_ux,valid_uy,valid_uz,valid_p)), batch_size=batch_size,
                                          shuffle=False, collate_fn=my_collate)

################################################################
# training and evaluation
################################################################
model = FNO3d(modes1, modes2, modes3, width).cuda()
print(count_params(model))

optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

myloss = LpLoss(size_average=False)

train_losses = []
test_losses = []
train_losses = []
train_losses_ux = []
train_losses_uy = []
train_losses_uz = []
train_losses_p = []
valid_losses = []
valid_losses_ux = []
valid_losses_uy = []
valid_losses_uz = []
valid_losses_p = []

for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_l2 = 0
    train_l2_ux = 0
    train_l2_uy = 0
    train_l2_uz = 0
    train_l2_p = 0
    for g,ui,x,y,z,ux,uy,uz, p in train_loader:
        g,ui,x,y,z,ux,uy,uz, p= g[0].cuda(),ui[0].cuda(),x[0].cuda(),y[0].cuda(),z[0].cuda(),ux[0].cuda(),uy[0].cuda(),uz[0].cuda(), p[0].cuda()
        g = g.view(batch_size, g.size(-3), g.size(-2), g.size(-1), 1)
        ui = ui.view(batch_size, ui.size(-3), ui.size(-2), ui.size(-1), 1)
        x = x.view(batch_size, x.size(-3), x.size(-2), x.size(-1), 1)
        y = y.view(batch_size, y.size(-3), y.size(-2), y.size(-1), 1)
        z = z.view(batch_size, z.size(-3), z.size(-2), z.size(-1), 1)
        input = torch.cat((g, ui,x,y,z), dim=-1)
        mask = g.clone()
        mask = torch.cat([mask, mask, mask, mask], dim=-1)

        optimizer.zero_grad()
        out = model(input)
        out = out*mask

        loss_ux = myloss(out[:, :, :, :, 0].view(batch_size, -1), ux.contiguous().view(batch_size, -1))
        loss_uy = myloss(out[:, :, :, :, 1].view(batch_size, -1), uy.contiguous().view(batch_size, -1))
        loss_uz = myloss(out[:, :, :, :, 2].view(batch_size, -1), uz.contiguous().view(batch_size, -1))
        loss_p = myloss(out[:, :, :, :, 3].view(batch_size, -1), p.contiguous().view(batch_size, -1))
        loss = loss_ux+loss_uy+loss_uz+loss_p
        loss.backward()

        optimizer.step()
        train_l2 += loss.item()
        train_l2_ux += loss_ux.item()
        train_l2_uy += loss_uy.item()
        train_l2_uz += loss_uz.item()
        train_l2_p += loss_p.item()

    scheduler.step()
    model.eval()
    test_l2_ux = 0.0
    test_l2_uy = 0.0
    test_l2_uz = 0.0
    test_l2_p = 0.0
    test_l2 = 0.0
    with torch.no_grad():
        for g, ui, x, y, z, ux, uy, uz, p in valid_loader:
            g, ui, x, y, z, ux, uy, uz, p = g[0].cuda(), ui[0].cuda(), x[0].cuda(), y[0].cuda(), z[0].cuda(), ux[
                0].cuda(), uy[0].cuda(), uz[0].cuda(), p[0].cuda()  # batch=1
            g = g.view(batch_size, g.size(-3), g.size(-2), g.size(-1), 1)
            ui = ui.view(batch_size, ui.size(-3), ui.size(-2), ui.size(-1), 1)
            x = x.view(batch_size, x.size(-3), x.size(-2), x.size(-1), 1)
            y = y.view(batch_size, y.size(-3), y.size(-2), y.size(-1), 1)
            z = z.view(batch_size, z.size(-3), z.size(-2), z.size(-1), 1)

            input = torch.cat((g, ui,x,y,z), dim=-1)
            mask = g.clone()
            mask = torch.cat([mask, mask, mask, mask], dim=-1)
            out = model(input)
            out2 = out * mask

            test_l2_ux += myloss(out2[:, :, :, :, 0].view(batch_size, -1), ux.contiguous().view(batch_size, -1)).item()
            test_l2_uy += myloss(out2[:, :, :, :, 1].view(batch_size, -1), uy.contiguous().view(batch_size, -1)).item()
            test_l2_uz += myloss(out2[:, :, :, :, 2].view(batch_size, -1), uz.contiguous().view(batch_size, -1)).item()
            test_l2_p += myloss(out2[:, :, :, :, 3].view(batch_size, -1), p.contiguous().view(batch_size, -1)).item()
            test_l2 = test_l2_ux + test_l2_uy + test_l2_uz + test_l2_p

    train_l2 /= ntrain
    test_l2 /= nvalid
    train_l2_ux /= ntrain
    train_l2_uy /= ntrain
    train_l2_uz /= ntrain
    train_l2_p /= ntrain
    test_l2_ux /= nvalid
    test_l2_uy /= nvalid
    test_l2_uz /= nvalid
    test_l2_p /= nvalid

    train_losses.append(train_l2)
    valid_losses.append(test_l2)
    train_losses_ux.append(train_l2_ux)
    train_losses_uy.append(train_l2_uy)
    train_losses_uz.append(train_l2_uz)
    train_losses_p.append(train_l2_p)
    valid_losses_ux.append(test_l2_ux)
    valid_losses_uy.append(test_l2_uy)
    valid_losses_uz.append(test_l2_uz)
    valid_losses_p.append(test_l2_p)

    t2 = default_timer()
    print(ep, t2 - t1, train_l2, test_l2)
    if ep % 50 == 0:
        train_losses1 = np.array(train_losses)
        valid_losses1 = np.array(valid_losses)
        scipy.io.savemat(savepath + '/loss_valid.mat',
                         {'train_losses': train_losses1, 'valid_losses_c1': valid_losses1})

        torch.save(model.state_dict(), savepath + "/fourier.pt")
        torch.save(optimizer.state_dict(), savepath + "/optimizer.pt")
        torch.save(scheduler.state_dict(), savepath + "/scheduler.pt")


train_losses_ux1=np.array(train_losses_ux)
train_losses_uy1=np.array(train_losses_uy)
train_losses_uz1=np.array(train_losses_uz)
train_losses_p1=np.array(train_losses_p)
scipy.io.savemat(savepath + '/loss_train4_24.mat', {'train_losses_ux1': train_losses_ux1, 'train_losses_uy1': train_losses_uy1, 'train_losses_uz1': train_losses_uz1, 'train_losses_p1': train_losses_p1})

valid_losses_ux1=np.array(valid_losses_ux)
valid_losses_uy1=np.array(valid_losses_uy)
valid_losses_uz1=np.array(valid_losses_uz)
valid_losses_p1=np.array(valid_losses_p)
scipy.io.savemat(savepath + '/loss_valid4_24.mat', {'valid_losses_ux1': valid_losses_ux1, 'valid_losses_uy1': valid_losses_uy1, 'valid_losses_uz1': valid_losses_uz1, 'valid_losses_p1': valid_losses_p1})

torch.save(model.state_dict(), savepath + "/fourier.pt")


#保存误差
train_losses1 = np.array(train_losses)
valid_losses1 = np.array(valid_losses)
scipy.io.savemat(savepath + '/loss.mat', {'train_losses1': train_losses1, 'valid_losses1': valid_losses1})

#画图
plt.clf()
plt.figure(1)
plt.xlabel('epoches')
plt.title('Model loss')

plt.plot(np.arange(len(train_losses)), train_losses, label="train Lploss")
plt.plot(np.arange(len(valid_losses)), valid_losses, label="valid Lploss")
plt.legend()  # 显示图例


plt.savefig(savepath + "/loss.jpg", dpi=300)
plt.show()
plt.close()

