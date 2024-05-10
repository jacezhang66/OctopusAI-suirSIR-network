from pathlib import Path
from torch.autograd import Variable
import torch.utils.data as data
from PIL import Image
from adamp import AdamP
from Network_New import suirSIR
from dataloader_util import TestData
from option import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
print("ROOT:",ROOT)

def _get_available_devices(n_gpu):
    sys_gpu = torch.cuda.device_count()
    if sys_gpu == 0:
        print('No GPUs detected, using the CPU')
        n_gpu = 0
    elif n_gpu > sys_gpu:
        print(f'Nbr of GPU requested is {n_gpu} but only {sys_gpu} are available')
        n_gpu = sys_gpu
    device = torch.device('cuda:0' if n_gpu > 0 else 'cpu')
    available_gpus = list(range(n_gpu))
    return device, available_gpus

device, available_gpus = _get_available_devices(opt.gpus)

model_root = ROOT / 'trained_models' / 'Best.pth'
input_root = ROOT / 'Test_dir' / 'temp_test'
save_path = ROOT / 'Result_dir' / 'temp_test'

if not os.path.exists(save_path):
    os.makedirs(save_path)

checkpoint = torch.load(model_root)
Mydata_ = TestData(input_root)
data_load = data.DataLoader(Mydata_, batch_size=1)

model = suirSIR().cuda()
model = torch.nn.DataParallel(model, device_ids=available_gpus)
optimizer = AdamP(model.parameters(), lr=2e-4, betas=(0.9, 0.999), weight_decay=1e-4)
model.load_state_dict(checkpoint['state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_dict'])
epoch = checkpoint['epoch']
model.eval()
print('START!')
print('Load model successfully!')
for data_idx, data_ in enumerate(data_load):
    data_input = data_
    data_input = Variable(data_input).cuda()
    print(data_idx)
    with torch.no_grad():
        result, result1, T, tb, _, A = model(data_input)
        name = Mydata_.A_paths[data_idx].split("\\")[-1]
        print(name)
        temp_res = np.transpose(result[0, :].cpu().detach().numpy(), (1, 2, 0))
        temp_res[temp_res > 1] = 1
        temp_res[temp_res < 0] = 0
        temp_res = (temp_res * 255).astype(np.uint8)
        temp_res = Image.fromarray(temp_res)
        temp_res.save('%s/%s' % (save_path, name))
        print('result saved!')

print('finished!')
