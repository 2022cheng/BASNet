import os
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms#, utils
# import torch.optim as optim

import numpy as np
from PIL import Image
import glob

# 导入自定义数据加载器和模型
from data_loader import RescaleT
from data_loader import CenterCrop
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model import BASNet
# 这段代码使用了基于深度学习的图像分割算法，具体是基于深度卷积神经网络（CNN）的图像分割模型，被命名为 BASNet。以下是代码中涉及到的主要部分：
#
# 数据加载和预处理：通过自定义的数据加载器 SalObjDataset 加载测试图像，并应用了一系列预处理操作，如尺度调整、转换为张量等，以便于模型输入。
# 模型定义和加载：使用了一个名为 BASNet 的模型，该模型在 model.py 文件中定义。在主函数中加载了预训练的 BASNet 模型权重。
# 推断过程：对每张测试图像进行推断，即将图像输入 BASNet 模型中，获取模型的输出结果，然后对输出结果进行归一化处理，并将处理后的结果保存到指定的目录中。
# 归一化函数：normPRED 函数用于将模型输出结果进行归一化处理，确保像素值在 0 到 1 之间

# 函数：将预测值归一化
def normPRED(d):
	ma = torch.max(d)
	mi = torch.min(d)

	dn = (d-mi)/(ma-mi)

	return dn

def save_output(image_name,pred,d_dir):

	predict = pred
	predict = predict.squeeze()
	predict_np = predict.cpu().data.numpy()

	# 将张量预测值转换为PIL图像
	im = Image.fromarray(predict_np*255).convert('RGB')
	img_name = image_name.split("/")[-1]
	image = io.imread(image_name)
	imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR)

	pb_np = np.array(imo)

	aaa = img_name.split(".")
	bbb = aaa[0:-1]
	imidx = bbb[0]
	for i in range(1,len(bbb)):
		imidx = imidx + "." + bbb[i]

	imo.save(d_dir+imidx+'.png')


if __name__ == '__main__':
	# --------- 1. get image path and name ---------
	
	image_dir = './test_data/test_images/'
	prediction_dir = './test_data/test_results/'
	model_dir = './saved_models/basnet_bsi/basnet.pth'
	# model_dir = './saved_models/basnet_bsi1/basnet_bsi_itr_1_train_21.361286_tar_2.808475.pth'
	# model_dir = './saved_models/basnet_bsi1/basnet_bsi_itr_1_train_21.968140_tar_2.701368.pth'

	img_name_list = glob.glob(image_dir + '*.png')
	
	# --------- 2. dataloader ---------
	#1. dataload数据加载
	test_salobj_dataset = SalObjDataset(img_name_list = img_name_list, lbl_name_list = [],transform=transforms.Compose([RescaleT(256),ToTensorLab(flag=0)]))
	test_salobj_dataloader = DataLoader(test_salobj_dataset, batch_size=1,shuffle=False,num_workers=1)
	
	# --------- 3. model define ---------定义模型
	print("...load BASNet...")
	net = BASNet(3,1)
	net.load_state_dict(torch.load(model_dir,map_location='cpu'))
	if torch.cuda.is_available():
		net.cuda()
	net.eval()
	
	# --------- 4. inference for each image ---------对每个图像进行推断
	for i_test, data_test in enumerate(test_salobj_dataloader):
	
		print("inferencing:",img_name_list[i_test].split("/")[-1])
	
		inputs_test = data_test['image']
		inputs_test = inputs_test.type(torch.FloatTensor)
	
		if torch.cuda.is_available():
			inputs_test = Variable(inputs_test.cuda())
		else:
			inputs_test = Variable(inputs_test)
	
		d1,d2,d3,d4,d5,d6,d7,d8 = net(inputs_test)
	
		# normalization 归一化
		pred = d1[:,0,:,:]
		pred = normPRED(pred)
	
		# save results to test_results folder
		save_output(img_name_list[i_test],pred,prediction_dir)
	
		del d1,d2,d3,d4,d5,d6,d7,d8
