import cv2
import numpy as np
import scipy.misc
import torch
#from Unet_SKA2 import UnetSKA2
from scipy.ndimage.interpolation import zoom

import h5py
import SimpleITK as sitk
import os
from PIL import Image
import nibabel as nib
import imageio

def colormap_cityscapes(n):
    cmap=np.zeros([n, 3]).astype(np.uint8)
    cmap[0] = np.array([0, 0,0])
    cmap[1] = np.array([255, 0,0])
    cmap[2] = np.array([ 255, 255, 0])
    cmap[3] = np.array([ 0,255,255])
    cmap[4] = np.array([ 255,0,255])
    cmap[5] = np.array([ 0,255,0])

    cmap[6] = np.array([ 0,0, 255])
    cmap[7] = np.array([ 153,0,  255])
    cmap[8] = np.array([ 74,134,232])

    
    return cmap
class Colorize:

    def __init__(self, n=9):
        #self.cmap = colormap(256)
        self.cmap = colormap_cityscapes(n)   #cmap是颜色表
        #self.cmap[n] = self.cmap[0] #把最后一类的颜色表设为[0,0,0]
        self.cmap = torch.from_numpy(self.cmap[:n]) # cmap由nump数组转为tensor

    def __call__(self, gray_image):
        gray_image=torch.from_numpy(gray_image).unsqueeze(0)
        size = gray_image.size()  # 网络output的大小
        color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0) 
        #生成3通道的image模板
        
        for label in range(0, len(self.cmap)):# 依次遍历label的颜色表
            mask = gray_image[0] == label  
            #gray_image[0] 是将三维的图像，以【1, 10, 10】为例，变成二维【10,10】,这个参数是外部传入，这里确保是二维单通道就行了
            #gray_image[0] == label 意思是将 gray_image[0]中为label值的元素视为true或者1，其他的元素为False 或0，得到mask的布尔图
            
            color_image[0][mask] = self.cmap[label][0] #取取颜色表中为label列表(【a,b,c】)的a
            #color_image[0]是取三通道模板中的单通道 ，然后把mask放上去
            color_image[1][mask] = self.cmap[label][1]  # 取b
            color_image[2][mask] = self.cmap[label][2]#   取c
        #print(color_image.size())
        #return color_image.numpy()
        return np.transpose(color_image.numpy(),(1,2,0))
def show_test(base_path="../dataset/prediction/",save_path="../dataset/Synapse/test_visualization"):
    
    if not os.path.exists(save_path):
        os.makedirs(save_path+"/pred")
        os.makedirs(save_path+"/label")
        os.makedirs(save_path+"/image")
    
    
    pred_list=glob(os.path.join(base_path,"*_pred.nii.gz"))
    pred_list.sort(reverse=True)
    label_list=glob(os.path.join(base_path,"*_gt.nii.gz"))
    label_list.sort(reverse=True)
    volums_list=glob(os.path.join(base_path,"*_img.nii.gz"))
    volums_list.sort(reverse=True)
    #print(pred_list)
    #print(label_list)
    #print(volums_list)
    assert len(pred_list)==len(label_list)
    assert len(volums_list)==len(pred_list)
    for i in range(len(pred_list)):
        pre=sitk.ReadImage(pred_list[i], sitk.sitkInt8)
        pre=sitk.GetArrayFromImage(pre)
        gt=sitk.ReadImage(label_list[i], sitk.sitkInt8)
        gt=sitk.GetArrayFromImage(gt)
        #volums=sitk.ReadImage(volums_list[i], sitk.sitkInt16) #    千万别用 sitk取读取image 会导致image 全黑或者 只有器官部分
        #volums=sitk.GetArrayFromImage(volums)
        volums=nib.load(volums_list[i])
        volums=volums.get_fdata()
        assert len(pre[0])==len(gt[0])
        
        assert len(pre.shape)==3
        x,y=pre.shape[1],pre.shape[2]
        for ind in range(pre.shape[0]):
            pre_slice = zoom(pre[ind,:,:], (224 / x, 224 / y), order=0)  
            gt_slice= zoom(gt[ind,:,:], (224 / x, 224 / y), order=0)
            volums_slice=volums[:,:,ind].transpose(1,0)
            imageio.imsave(save_path+"/pred/"+pred_list[i].split("/")[-1].split(".")[0].split("_")[0]+"_slice{}.png".format(ind),Colorize(9)(pre_slice))
            
            imageio.imsave(save_path+"/label/"+label_list[i].split("/")[-1].split(".")[0].split("_")[0]+"_slice{}.png".format(ind),Colorize(9)(gt_slice))
            imageio.imwrite(save_path+"/image/"+volums_list[i].split("/")[-1].split(".")[0].split("_")[0]+"_slice{}.png".format(ind),volums_slice)
        
            

   
if __name__=="__main__":
    #show_heatmap()
    show_test()
