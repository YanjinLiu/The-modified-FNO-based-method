import torch
from fnomodel import FNO3d
from utilities3 import *
from timeit import default_timer

#Model parameters
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


#Import the pre-trained model
model_path = 'fourier_c.pt'#
model_data = torch.load(model_path)
model = FNO3d(modes1, modes2, modes3, width).cuda()
model.load_state_dict(model_data)
myloss = LpLoss(size_average=False)
#print(model)
savepath='test/pred/c'#

#load data
PATH = "sort_D_a/"
TRAIN_PATH1 = PATH+'input_a_c.mat'
TRAIN_PATH2 = PATH+'input_ui_c.mat'
TRAIN_PATH3 = PATH+'input_x1.mat'
TRAIN_PATH4 = PATH+'input_y1.mat'
TRAIN_PATH5 = PATH+'input_z1.mat'
TRAIN_PATH6 = PATH+'input_c.mat'
TRAIN_PATH7 = PATH+'input_c0.mat'

myloss = LpLoss(size_average=False)

reader1 = MatReader(TRAIN_PATH1)
reader2 = MatReader(TRAIN_PATH2)
reader3 = MatReader(TRAIN_PATH3)
reader4 = MatReader(TRAIN_PATH4)
reader5 = MatReader(TRAIN_PATH5)
reader6 = MatReader(TRAIN_PATH6)
reader7 = MatReader(TRAIN_PATH7)

#
input_a = reader1.data['input_a_c']
input_ui = reader2.data['input_ui']
input_x = reader3.data['input_x']
input_y = reader4.data['input_y']
input_z = reader5.data['input_z']
input_c = reader6.data['input_c']
input_c0 = reader7.data['input_c0']

input_a=input_a.reshape((input_a.size))
input_ui=input_ui.reshape((input_ui.size))
input_x=input_x.reshape((input_x.size))
input_y=input_y.reshape((input_y.size))
input_z=input_z.reshape((input_z.size))
input_c=input_c.reshape((input_c.size))
input_c0=input_c0.reshape((input_c0.size))

for i in range(input_a.size):
    input_a[i] = torch.tensor(input_a[i][:, ::sub, ::sub, ::sub], dtype=torch.float).permute(3, 2, 1, 0)
    input_ui[i] = torch.tensor(input_ui[i][:, ::sub, ::sub, ::sub], dtype=torch.float).permute(3, 2, 1, 0)
    input_x[i] = torch.tensor(input_x[i][:, ::sub, ::sub, ::sub], dtype=torch.float).permute(3, 2, 1, 0)
    input_y[i] = torch.tensor(input_y[i][:, ::sub, ::sub, ::sub], dtype=torch.float).permute(3, 2, 1, 0)
    input_z[i] = torch.tensor(input_z[i][:, ::sub, ::sub, ::sub], dtype=torch.float).permute(3, 2, 1, 0)
    input_c[i] = torch.tensor(input_c[i][:, ::sub, ::sub, ::sub], dtype=torch.float).permute(3, 2, 1, 0)
    input_c0[i] = torch.tensor(input_c0[i][:, ::sub, ::sub, ::sub], dtype=torch.float).permute(3, 2, 1, 0)


test_a = input_a[ntrain+nvalid:ntotal]
test_ui = input_ui[ntrain+nvalid:ntotal]
test_x = input_x[ntrain+nvalid:ntotal]
test_y = input_y[ntrain+nvalid:ntotal]
test_z = input_z[ntrain+nvalid:ntotal]
test_c = input_c[ntrain+nvalid:ntotal]
test_c0 = input_c0[ntrain+nvalid:ntotal]



def my_collate(batch):
    train_a = [item[0] for item in batch]
    train_ui = [item[1] for item in batch]
    train_c0 = [item[2] for item in batch]
    train_x = [item[3] for item in batch]
    train_y = [item[4] for item in batch]
    train_z = [item[5] for item in batch]
    train_c = [item[6] for item in batch]
    return [train_a, train_ui, train_c0, train_x, train_y, train_z, train_c]

#test
pred = []
test_losses=[]
test_losses_ux=[]
test_losses_uy=[]
test_losses_uz=[]
test_losses_p=[]
test_losses_c=[]
index = 0
test_loader = torch.utils.data.DataLoader(list(zip(test_a, test_ui, test_c0, test_x, test_y, test_z, test_c)), batch_size=1,
                                           shuffle=False, collate_fn=my_collate)

with torch.no_grad():
    for g, ui, c0, x, y, z, c in test_loader:
        test_l3_c = 0.0
        test_l3 = 0
        t1 = default_timer()
        g, ui, c0, x, y, z, c = g[0].cuda(), ui[0].cuda(), c0[0].cuda(), x[0].cuda(), y[0].cuda(), z[0].cuda(), c[
            0].cuda()
        g = g.view(batch_size, g.size(-3), g.size(-2), g.size(-1), 1)
        ui = ui.view(batch_size, ui.size(-3), ui.size(-2), ui.size(-1), 1)
        c0 = c0.view(batch_size, c0.size(-3), c0.size(-2), c0.size(-1), 1)
        x = x.view(batch_size, x.size(-3), x.size(-2), x.size(-1), 1)
        y = y.view(batch_size, y.size(-3), y.size(-2), y.size(-1), 1)
        z = z.view(batch_size, z.size(-3), z.size(-2), z.size(-1), 1)

        input = torch.cat((g, ui, c0, x, y, z), dim=-1)
        mask = g.clone()
        stay = torch.ones(g.shape)
        stay = stay.cuda()

        out = model(input)
        out3 = out * mask

        test_l3_c += 100 * myloss(out3[:, :, :, :, 0].view(batch_size, -1), c.contiguous().view(batch_size, -1)).item()
        test_l3 = test_l3_c

        pred.append(out3)

        test_losses.append(test_l3)
        test_losses_c.append(test_l3_c)
        t2 = default_timer()

        print(index, t2-t1, test_l3)
        index = index + 1

#save results
pred1 = np.empty(len(pred),dtype=object)
for i in range(ntest):
    pred1[i]=pred[i].cpu().numpy()
scipy.io.savemat(savepath + '/pred_c.mat', {'pred_c': pred1})

test_losses1 = np.array(test_losses_c)
scipy.io.savemat(savepath+'/test_losses_c.mat', {'test_losses_c': test_losses1})