import torch
from fnomodel import FNO3d_up
from utilities3 import *
from timeit import default_timer

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
h = int(((90 - 1) / r) + 1)
s = int(((500 - 1) / r) + 1)
t = int(((15 - 1) / r) + 1)

#Import the pre-trained model
model_path = 'fourier.pt'#
model_data = torch.load(model_path)
model = FNO3d_up(modes1, modes2, modes3, width).cuda()
model.load_state_dict(model_data)
#print(model)
savepath='F:/FNO/test/pred/up/'#

#load data
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
myloss = LpLoss(size_average=False)

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

#test
pred = []
test_losses=[]
test_losses_ux=[]
test_losses_uy=[]
test_losses_uz=[]
test_losses_p=[]
index = 0
test_loader = torch.utils.data.DataLoader(list(zip(test_a, test_ui, test_x, test_y, test_z,test_ux,test_uy,test_uz,test_p)), batch_size=1,
                                           shuffle=False, collate_fn=my_collate)
with torch.no_grad():
    for g, ui, x, y, z in test_loader:
        test_l3_ux = 0.0
        test_l3_uy = 0.0
        test_l3_uz = 0.0
        test_l3_p = 0.0
        test_l3 = 0
        t1 = default_timer()
        g, ui, x, y, z, ux, uy, uz, p = g[0].cuda(), ui[0].cuda(), x[0].cuda(), y[0].cuda(), z[0].cuda(), ux[0].cuda(), \
            uy[0].cuda(), uz[0].cuda(), p[0].cuda()
        g = g.view(batch_size, g.size(-3), g.size(-2), g.size(-1), 1)
        ui = ui.view(batch_size, ui.size(-3), ui.size(-2), ui.size(-1), 1)
        x = x.view(batch_size, x.size(-3), x.size(-2), x.size(-1), 1)
        y = y.view(batch_size, y.size(-3), y.size(-2), y.size(-1), 1)
        z = z.view(batch_size, z.size(-3), z.size(-2), z.size(-1), 1)

        input = torch.cat((g, ui, x, y, z), dim=-1)
        mask = g.clone()
        mask = torch.cat([mask, mask, mask, mask], dim=-1)

        out = model(input)
        out3 = out * mask

        test_l3_ux += myloss(out3[:, :, :, :, 0].view(batch_size, -1), ux.contiguous().view(batch_size, -1)).item()
        test_l3_uy += myloss(out3[:, :, :, :, 1].view(batch_size, -1), uy.contiguous().view(batch_size, -1)).item()
        test_l3_uz += myloss(out3[:, :, :, :, 2].view(batch_size, -1), uz.contiguous().view(batch_size, -1)).item()
        test_l3_p += myloss(out3[:, :, :, :, 3].view(batch_size, -1), p.contiguous().view(batch_size, -1)).item()
        test_l3 = test_l3_ux + test_l3_uy + test_l3_uz + test_l3_p

        pred.append(out3)

        test_losses.append(test_l3)
        test_losses_ux.append(test_l3_ux)
        test_losses_uy.append(test_l3_uy)
        test_losses_uz.append(test_l3_uz)
        test_losses_p.append(test_l3_p)
        t2 = default_timer()

        print(index, t2-t1, test_l3)
        index = index + 1

pred1 = np.empty(len(pred),dtype=object)
for i in range(ntest):
    pred1[i]=pred[i].cpu().numpy()
scipy.io.savemat(savepath + 'pred_up.mat', {'pred_up': pred1})

test_losses1 = np.array(test_losses)
scipy.io.savemat(savepath+'test_losses_up.mat', {'test_losses_up': test_losses1})