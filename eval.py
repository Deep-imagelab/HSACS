import torch
import os
import numpy as np
import cv2
from utils.HSACS import HSACS
import glob
from utils.utils import reconstruction_patch_image_gpu, save_matv73

os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
model_path = 'model.pth'
result_path = 'results'
img_path = 'NTIRE2018_Validate_Clean'
var_name = 'rad'

# save results
if not os.path.exists(result_path):
    os.makedirs(result_path)
model = HSACS()
save_point = torch.load(model_path)
model_param = save_point['state_dict']
model_dict = {}
for k1, k2 in zip(model.state_dict(), model_param):
    model_dict[k1] = model_param[k2]
model.load_state_dict(model_dict)
model = model.cuda()

img_path_name = glob.glob(os.path.join(img_path, '*.png'))
img_path_name.sort()

for i in range(len(img_path_name)):
    # load rgb images
    rgb = cv2.imread(img_path_name[i])
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    rgb = np.float32(rgb) / 255.0
    rgb = np.expand_dims(np.transpose(rgb, [2, 0, 1]), axis=0).copy()
    print(img_path_name[i].split('/')[-1])

    _, img_res = reconstruction_patch_image_gpu(rgb, model, 256, 256)
    _, img_res_overlap = reconstruction_patch_image_gpu(rgb[:, :, 256//2:, 256//2:], model, 256, 256)
    img_res[256//2:, 256//2:, :] = (img_res[256//2:, 256//2:, :] + img_res_overlap) / 2.0

    mat_name = img_path_name[i].split('/')[-1][:-10] + '.mat'
    mat_dir = os.path.join(result_path, mat_name)

    save_matv73(mat_dir, var_name, img_res)







