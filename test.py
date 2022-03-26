import numpy as np
import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader
from torch import optim
from torchvision.transforms import transforms
from PIL import Image
import logging
from networks.MobileNetV2_unet import MobileNetV2_unet
from networks.Unet_SKA2 import UnetSKA2
from networks.enet import ENet
from networks.erfnet import ERFNet
import os
from dataloader import *
import  time
import torch.nn.functional as F
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from networks.kd_model import NetModel
from networks.feature_projection import Paraphraser
from medpy import metric
import SimpleITK as sitk
from scipy.ndimage import zoom
import math
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
print(torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parse = argparse.ArgumentParser()


#logger使用

formatter = logging.Formatter("%(asctime)s:%(levelname)s:[%(filename)s]:%(message)s")




#test logger
logger_test = logging.getLogger("test")
logger_test.setLevel(logging.INFO)

fh_test = logging.FileHandler("test.log")
sh_test =logging.StreamHandler()

sh_test.setLevel(logging.INFO)
fh_test.setLevel(logging.INFO)

fh_test.setFormatter(formatter)
sh_test.setFormatter(formatter)

logger_test.addHandler(fh_test)
logger_test.addHandler(sh_test)

class Evaluator(object):
    """
       Class to calculate mean-iou using fast_hist method
    """

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes , num_classes))

    def _fast_hist(self, label_pred , label_true):
        # confusion matrix
        #mask 会变成一个label_truec长度的向量,label_true[mask]只保留mask中所有值为True的部分
        mask = (label_true >= 0) & (label_true < self.num_classes)
        
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) +
            label_pred[mask], minlength=self.num_classes**2).reshape(self.num_classes, self.num_classes)
        return hist

    def add_batch(self, predictions,ground_truth):
        for lp, lt in zip(predictions,ground_truth):
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())

    def evalute(self):
        acc = np.diag(self.hist).sum()/self.hist.sum()
        acc_cls = np.diag(self.hist)/self.hist.sum(axis=1)
        mean_acc_cls = np.nanmean(acc_cls)
        iou = np.diag(self.hist)/(self.hist.sum(axis=1)+self.hist.sum(axis=0)-np.diag(self.hist))
        mean_iou = np.nanmean(iou)
        freq = self.hist.sum(axis=1)/self.hist.sum()
        frwvacc = (freq[freq > 0]*iou[freq > 0]).sum()
        TP = self.hist[1,1]
        TN = self.hist[0,0]
        FP = self.hist[0,1]
        FN = self.hist[1,0]
        #print("TP:{},TN:{},FP:{},FN:{}".format(TP,TN,FP,FN))
        #print(self.hist)
        Pre = TP / (TP + FP) #presicison 正确预测样本占全部预测的比例 预测结果中，某一类被正确预测的概率
        ACC = (TP+TN) / (TP+TN+FN+FP)
        Recall  = TP / (TP+FN)  #recall 正确预测样本占groundtruth的比列 真实值中，某一类被预测正确的概率
        TNR = TN/(TN+FP)  #TNR  true negative rate 描述识别出的负例占所有负例的比例 特异度
        F1  = 2*Pre*Recall/(Pre+Recall)
        return Pre,ACC, Recall, TNR,mean_iou, iou, F1

class Inference(object):
    def __init__(self, model, num_classes, test_image):
        self.model = model
        self.num_classes = num_classes
        b, c, h, w = test_image.size()
        self.b, self.h, self.w = b, h, w
        self.test_image = test_image
    def single_inference(self, test_image):
        middle_feature1,middle_feature,pre =self.model(test_image)
        #pre = F.interpolate(pre, size=(self.h, self.w), mode="bilinear", align_corners=True)
        pre = F.softmax(pre, dim=1)
        pre = pre.data.cpu()
        return pre

    def flip_image(self, img):
        flipped = torch.flip(img,[2,3])
        return flipped

    def fushion_avg(self, pre):
        pre_final = torch.seros(self.b, self.num_classes, self.h, self.w)
        for pre_scaled in pre:
            pre_final = pre_final+pre_scaled
        pre_final = pre_final/len(pre)
        return pre_final

    def mutliscale_inference(self, test_image, is_Flip=True):
        pre = []
        inf_scales = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25]
        for scale in inf_scales:
            img_scaled = F.interpolate(test_image, size=(int(self.h*scale), int(scale.w*scale)),mode="bilinear",align_corners=True)
            pre_scaled = self.single_inference(img_scaled)
            pre.append(pre_scaled)
            if is_Flip:
                img_scaled = self.flip_image(img_scaled)
                pre_scaled = self.single_inference(img_scaled)
                pre_scaled = self.flip_image(pre_scaled)
                pre.append(pre_scaled)
        pre_final = self.fushion_avg(pre)
        return pre_final

def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0
    else:
        return 0, 0


def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                _,_,outputs = net(input)
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            _,_,outputs = net(input)
            out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    if test_save_path is not None:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.nii.gz")
    return metric_list
    
    
def test(network="MobileNetV2_unet",pth="010.pth",types="sp",num_classes=2):
    assert num_classes==2
    state_dict = torch.load(os.path.join("./models/GastricCancer/student_"+types,pth))
    if network=="MobileNetV2_unet":
        model= MobileNetV2_unet(num_classes=num_classes,pre_trained=None)
    elif network=="ENet":
        model=ENet(num_classes)
    elif network=="ERFNet":
        model=ERFNet(num_classes)
    elif network=="UnetSKA2":
        model=UnetSKA2(in_ch=1,out_ch=num_classes)
    
    #model =UnetSKA2(in_ch=1,out_ch=2)

    if torch.cuda.device_count() > 1:  
        model = nn.DataParallel(model)
   
    model.load_state_dict(state_dict)
    logger_test.info('Model loaded from {}'.format(pth))
    
    model.to(device)
    
    custom_dataset = GastricCancerDataset(image_path="dataset/GastricCancer/test", transform=transforms.Compose([ToTensor()]),type_image="resize")
    data_loader = DataLoader(custom_dataset, batch_size=1, shuffle=True)
    
    
    
    evalutor = Evaluator(num_classes=num_classes)
    model.eval()
    
    with torch.no_grad():
        for sample in data_loader:
            image = sample['image']
            label = sample['label']
            name = sample['name']
            image = image.to(device)
            label = label.to(device)
            
            pre = Inference(model, num_classes, image).single_inference(image)
            
            
            pre = pre[:, :, 95:95+64, 95:95+64]
            label = label[ :, 95:95+64, 95:95+64]
                
            
            pre = torch.argmax(pre, dim=1)
            pre = pre.cpu().numpy()
            label = label.cpu().numpy()
            evalutor.add_batch(pre, label)
            pre=pre.squeeze()
            pre = pre*255
            
            pre = Image.fromarray(np.uint8(pre))
            if not os.path.exists("./dataset/prediction"):
              os.mkdir("./dataset/prediction")
            pre.save("./dataset/prediction/%s"%name[0].split("/")[-1]+".png")
        Pre,ACC, Recall, TNR, mIOU,iou, F1 = evalutor.evalute()
        logger_test.info("Pre:{} ACC:{} Recall:{} TNR:{} mIOU:{} iou:{} F1:{}".format(Pre,ACC, Recall, TNR,mIOU, iou, F1))
    return ACC,mIOU
def test_Synapse(network="MobileNetV2_unet",pth="010.pth",types="sp",num_classes=9):   
    assert num_classes==9
    state_dict = torch.load(os.path.join("./models/Synapse/student_"+types,pth))
    
    if network=="MobileNetV2_unet":
        model= MobileNetV2_unet(num_classes=num_classes,pre_trained=None)
    elif network=="ENet":
        model=ENet(num_classes)
    elif network=="ERFNet":
        model=ERFNet(num_classes)
    elif network=="UnetSKA2":
        model=UnetSKA2(in_ch=1,out_ch=num_classes)
    
    if torch.cuda.device_count() > 1:  
        model = nn.DataParallel(model)
    model.load_state_dict(state_dict)
    logger_test.info('Model loaded from {}'.format(pth))
    model.to(device)
    custom_dataset = Synapsedataset(base_dir="dataset/Synapse",list_dir="dataset/Synapse/lists_Synapse",split="test_vol")
    data_loader = DataLoader(custom_dataset, batch_size=1, shuffle=True)
    model.eval()
    metric_list = 0.0
    if not os.path.exists("./dataset/prediction"):
        os.mkdir("./dataset/prediction")
    
    for sample in data_loader:
        image = sample['image']
        label = sample['label']
        name = sample['name']
        metric_i = test_single_volume(image, label, model, classes=num_classes, patch_size=[224,224],
                                      test_save_path="./dataset/prediction", case=name[0], z_spacing=1)
        metric_list += np.array(metric_i)
    metric_list = metric_list / len(custom_dataset)
    for i in range(1, num_classes):
        logger_test.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i-1][0], metric_list[i-1][1]))
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    logger_test.info('Testing performance in model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
    return performance,mean_hd95 


def main():
    parse = argparse.ArgumentParser()
    parse.add_argument("-m", "--mode", type=str, default="test", help="inference mode(default test)")
    parse.add_argument("-b", "--batch_size", type=int, default=8, help="batch_size")
    parse.add_argument("--student",type=str,default="MobileNetV2_unet")
    parse.add_argument("--teacher",type=str,default="FANet")
    parse.add_argument("-n", "--num_classes", type=int, default=2)
    
    parse.add_argument("-p", "--pth", default=False)
    
    parse.add_argument("-t","--types", default="csc")
    
    parse.add_argument("-a","--adversial",type=bool,default=False)
    parse.add_argument("--dataset",type=str,default="GastricCancer")
    
    args = parse.parse_args()
    assert args.mode in [ "test"]
    if args.mode=="test":
        if args.dataset=="GastricCancer":
            if args.pth:
            
                test(network=args.student,pth=args.pth,types=args.types,num_classes=args.num_classes)
            else:
                logger.warning("请输入指定加载模型")
        else:
            if args.pth:
                test_Synapse(network=args.student,pth=args.pth,types=args.types,num_classes=args.num_classes)
                
            else:
                logger.warning("请输入指定加载模型")
if __name__ == "__main__":
    main()
    #train_loop()
    