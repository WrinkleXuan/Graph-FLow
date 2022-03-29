"""
zwx
2021 4/16
"""

import numpy as np
import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader
from torch import optim
from torchvision.transforms import transforms
from PIL import Image
import logging
from networks.Unet import Unet
from networks.MobileNetV2_unet import MobileNetV2_unet
from networks.Unet_SKA2 import UnetSKA2
from networks.Unetpp import NestedUNet
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
from networks.deeplabv3_plus import  DeepLab
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from utils.criterion import CriterionParaphraser
from torch.utils.tensorboard import SummaryWriter
from torch.utils import data
from itertools import cycle
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
print(torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parse = argparse.ArgumentParser()

#logger使用

formatter = logging.Formatter("%(asctime)s:%(levelname)s:[%(filename)s]:%(message)s")

logger = logging.getLogger("train")
logger.setLevel(logging.INFO)
fh = logging.FileHandler("train.log")
sh =logging.StreamHandler()
sh.setLevel(logging.INFO)
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)
sh.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(sh)
#logging.basicConfig(filename="train.log",format="%(asctime)s:%(levelname)s:[%(filename)s]:%(message)s", level=logging.INFO)

#paraphraser logger
logger_para= logging.getLogger("para")
logger_para.setLevel(logging.INFO)

fh_para = logging.FileHandler("paraphraser.log")
sh_para =logging.StreamHandler()

sh_para.setLevel(logging.INFO)
fh_para.setLevel(logging.INFO)

fh_para.setFormatter(formatter)
sh_para.setFormatter(formatter)

logger_para.addHandler(fh_para)
logger_para.addHandler(sh_para)




def adjust_learning_rate_poly(optimizer, all_iter, now_iter, epoch):
    if epoch <= 50:
        #base_lr = 3*0.0003
        base_lr = 0.0003
        
    elif epoch>50 and epoch<=100:
        #base_lr = 3*0.00003
        base_lr=0.0003
    elif epoch>100 :
        #base_lr=3*0.00003
        base_lr = 0.000003

    lr = base_lr*((1-(now_iter/all_iter))**0.9)
    #lr = base_lr
    if(len(optimizer.param_groups)==1):
        optimizer.param_groups[0]["lr"] = lr
    else:
        optimizer.param_groups[0]['lr'] = lr*0.1
    for i in range(1,len(optimizer.param_groups)):
        optimizer.param_groups[i]['lr'] = lr


def train(**kwargs):
    writer=SummaryWriter(os.getcwd()+"/runs")
    batch_size = kwargs["batch_size"]
    num_classes = kwargs["num_classes"]
    learing_rate = kwargs["learing_rate"]
    epochs = kwargs["epochs"]
    teacher_pth = kwargs["pth"]
    lambda_s=kwargs["lambda_s"]
    lambda_d=kwargs["lambda_d"]
    lambda_adv=kwargs["lambda_adv"]
    types=kwargs["types"]
    adversial=kwargs["adversial"]
    KD=kwargs["KD"]
    adv_loss_type=kwargs["adv_loss_type"]
    dataset=kwargs["dataset"]
    student=kwargs["student"]
    teacher=kwargs["teacher"]
    if dataset == "GastricCancer":
        assert num_classes==2
    else:
        assert dataset=="Synapse"
        assert num_classes==9
    start_all=time.time()
    logger.info("         ---------------------------         ")
    logger.info("types == {} adversial == {} adv_loss_type == {} KD=={}".format(types,adversial,adv_loss_type,KD))
    logger.info("lambda_s == {}  lambda_d == {} lambda_adv == {}".format(lambda_s,lambda_d,lambda_adv))
    logger.info("Teacher == {}  Student == {}".format(teacher,student))
    
    model = NetModel(teacher,student,learing_rate,lr_d=2e-5,preprocess_GAN_mode=2,num_classes= num_classes,batch_size=batch_size,imsize_for_adv=256,adv_conv_dim=64,adv_loss_type=adv_loss_type\
                     ,adversial=adversial,KD=KD,student_pth=False, teacher_pth=teacher_pth,lambda_s=lambda_s,lambda_adv=lambda_adv,lambda_d=lambda_d,mode=types)
    
    if dataset=="GastricCancer":
        #custom_dataset = GastricCancerdataset(image_path="dataset/GastricCancer/train", transform=transforms.Compose([RandomHorizontalFlip(),
        #                                                                                       RandomVerticalFlip(),
        #                                                                                       ToTensor()]),type_image="resize")
        custom_dataset_sup = GastricCancerDataset_Semi(image_path="dataset/GastricCancer/resize",split="train",supervised=True, percent_labeled=0.6,transform=transforms.Compose([RandomHorizontalFlip(),RandomVerticalFlip(),ToTensor()]))
        custom_dataset_unsup = GastricCancerDataset_Semi(image_path="dataset/GastricCancer/resize",split="train",supervised=False, percent_labeled=0.6,transform=transforms.Compose([RandomHorizontalFlip(),RandomVerticalFlip(),ToTensor()]))
        
    elif dataset=="Synapse":
        #custom_dataset = Synapsedataset(base_dir="dataset/Synapse",list_dir="dataset/Synapse/lists_Synapse",split="train",\
        #                                                transform=transforms.Compose([RandomGenerator(output_size=[224,224])]))
        custom_dataset_sup = Synapsedataset_Semi(base_dir="dataset/Synapse",list_dir="dataset/Synapse/lists_Synapse",split="train",\
                                                        supervised=True,num_case_labled=2,transform=transforms.Compose([RandomGenerator(output_size=[224,224])]))
        custom_dataset_unsup = Synapsedataset_Semi(base_dir="dataset/Synapse",list_dir="dataset/Synapse/lists_Synapse",split="train",\
                                                        supervised=False,num_case_labled=2,transform=transforms.Compose([RandomGenerator(output_size=[224,224])]))
        
        
    supervised_loader = DataLoader(custom_dataset_sup, batch_size=batch_size//2, shuffle=True, drop_last=True, num_workers=0)
    unsupervised_loader = DataLoader(custom_dataset_unsup, batch_size=batch_size//2, shuffle=True, drop_last=True, num_workers=0)
    
    if len(supervised_loader.dataset)>len(unsupervised_loader):
        #data_loader=zip(supervised_loader,cycle(unsupervised_loader))
        dt_size=len(supervised_loader.dataset)
    else:
        #data_loader=zip(cycle(supervised_loader),unsupervised_loader)
    
        dt_size = len(unsupervised_loader.dataset)
    
    
    #dt_size=len(supervised_loader.dataset)
    
    all_iter = math.ceil(dt_size/(batch_size//2))*epochs
    save_path="./models"+"/"+dataset+"/student_"+types
    logger.info("Start Training: Total epochs:{} Total iterations:{} Batch_Size:{} Dataset:{} Labeled Training_Size:{} UnLabeled Training_Size:{} device:{}".\
             format(epochs,all_iter, batch_size,dataset, len(supervised_loader.dataset),len(unsupervised_loader.dataset),device))
    #logger.info("Start Training: Total epochs:{} Total iterations:{} Batch_Size:{} Dataset:{} Labeled Training_Size:{}  device:{}".\
    #           format(epochs,all_iter, batch_size,dataset, len(supervised_loader.dataset),device))
    
    for epoch in range(epochs):
        logger.info("Epoch = {}".format(epoch+1))
        start = time.time()
        epoch_loss = 0
        epoch_kd_loss=0
        epoch_ce_loss = 0
        epoch_sp_loss = 0
        epoch_ch_loss = 0
        epoch_sc_loss = 0
        epoch_sc_loss_encoder = 0
        epoch_sc_loss_decoder = 0
        epoch_csc_loss=0
        epoch_graph_edges_loss=0
        epoch_graph_vertices_loss=0
        if types=="IFVD":
            epoch_IFVD_loss=0
        if types =="SKD":
            epoch_pi_loss=0
            epoch_pair_loss=0
        if types =="AT":
            epoch_at_loss=0
        if types =="FSP":
            epoch_fsp_loss=0
        epoch_adv_G_loss=0
        
        epoch_w_distance=0
        step = 0
        
        if len(supervised_loader.dataset)>len(unsupervised_loader):
            data_loader=zip(supervised_loader,cycle(unsupervised_loader))
        else:
            data_loader=zip(cycle(supervised_loader),unsupervised_loader)
        
        
        for sample in data_loader:
        #for sample in supervised_loader:
            tmp_start = time.time()
            step += 1
            now_iter = step+epoch*math.ceil(dt_size/batch_size)

            model.optimize_parameters(data=sample, lambda_kd=1, all_iter=all_iter, now_iter=now_iter, epoch=epoch+1)
            epoch_loss += model.loss
            epoch_ce_loss+=model.ce_loss
            
            
            writer.add_scalar("Total Loss",model.loss,now_iter)
            writer.add_scalar("CE Loss",model.ce_loss,now_iter)
            #writer.add_scalar("KD Loss",model.kd_loss,now_iter)
            if KD==True:
                epoch_kd_loss+=model.kd_loss
                writer.add_scalar("KD Loss",model.kd_loss,now_iter)
            if types=="sp":
                epoch_sp_loss+=model.sp_loss
            elif types=="sc":
                epoch_sc_loss_encoder+=model.sc_loss_encoder
                epoch_sc_loss_decoder+=model.sc_loss_decoder
            elif types=="csc":
                epoch_csc_loss+=model.csc_loss
                #epoch_sc_loss_encoder+=model.sc_loss_encoder
                #epoch_sc_loss_decoder+=model.sc_loss_decoder
            elif types=="IFVD":
                epoch_IFVD_loss+=model.IFVD_loss
            elif types=="SKD":
                epoch_pi_loss+=model.pi_loss
                epoch_pair_loss+=model.pair_loss
            elif types=="AT":
                epoch_at_loss+=model.at_loss
            elif types=="FSP":
                epoch_fsp_loss+=model.fsp_loss
            elif types=="Graph":
                epoch_graph_edges_loss+=model.graph_edges_loss
                epoch_graph_vertices_loss+=model.graph_vertices_loss
                writer.add_scalar("Graph Edges Loss",model.graph_edges_loss,now_iter)
                writer.add_scalar("Graph Vertices Loss",model.graph_vertices_loss,now_iter)
            if(adversial==True):
                epoch_adv_G_loss+=model.adv_G_loss
                epoch_w_distance+=model.w_distance
                writer.add_scalar("Adv G Loss",model.adv_G_loss,now_iter)
                writer.add_scalar("w_distance",model.w_distance,now_iter)
                
            tmp_end = time.time()
        end = time.time()
        if types=="sp":
            logger.info("第{}个Epoch:total loss={} CrossEntropyLoss = {} SP loss ={} ".format(epoch+1, epoch_loss/step,epoch_ce_loss/step,epoch_sp_loss/step))
        elif types=="sc":
            logger.info("第{}个Epoch:total loss={} CrossEntropyLoss = {} spatial&&Channel encoder loss ={} spatial&&Channel decoder loss ={} ".format(epoch+1, epoch_loss/step,epoch_ce_loss/step,epoch_sc_loss_encoder/step,epoch_sc_loss_decoder/step))
                                                                                                                            
        elif types=="csc":
            logger.info("第{}个Epoch:total loss={} CrossEntropyLoss = {} CSC loss ={} spatial&&Channel encoder loss ={} spatial&&Channel decoder loss ={}".format(epoch+1, epoch_loss/step,epoch_ce_loss/step,epoch_csc_loss/step,
                        epoch_sc_loss_encoder/step,epoch_sc_loss_decoder/step))
        elif types=="IFVD":
            logger.info("第{}个Epoch:total loss={} CrossEntropyLoss = {} IFVD loss ={} ".format(epoch+1, epoch_loss/step,epoch_ce_loss/step,epoch_IFVD_loss/step))
        elif types=="SKD":
            logger.info("第{}个Epoch:total loss={} CrossEntropyLoss = {} PI loss ={} Pair loss={} ".format(epoch+1, epoch_loss/step,epoch_ce_loss/step,epoch_pi_loss/step,epoch_pair_loss/step))
        elif types=="AT":
            logger.info("第{}个Epoch:total loss={} CrossEntropyLoss = {} AT loss ={} ".format(epoch+1, epoch_loss/step,epoch_ce_loss/step,epoch_at_loss/step))
        elif types=="FSP":
            logger.info("第{}个Epoch:total loss={} CrossEntropyLoss = {} FSP loss ={} ".format(epoch+1, epoch_loss/step,epoch_ce_loss/step,epoch_fsp_loss/step))
        elif types=="Graph":
            logger.info("第{}个Epoch:total loss={} CrossEntropyLoss = {} Graph Edegs loss ={} Graph Vertices loss ={} KD loss={}".format(epoch+1, epoch_loss/step,epoch_ce_loss/step,epoch_graph_edges_loss/step,epoch_graph_vertices_loss/step,epoch_kd_loss/step))
        else:
            logger.info("第{}个Epoch:total loss={} CrossEntropyLoss = {} KD loss={}".format(epoch+1, epoch_loss/step,epoch_ce_loss/step,epoch_kd_loss/step))
        
        if(adversial==True):
            
            logger.info("第{}个Epoch: Adversarial Generator loss={} w_distance = {} ".format(epoch+1, epoch_adv_G_loss/step,epoch_w_distance/step))
            if KD==True:
                logger.info("第{}个Epoch:KD loss={}".format(epoch+1, epoch_kd_loss/step)) 
        logger.info("traing time={}".format(end-start))

        
        
        if( (epoch + 1) % 5 == 0):
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            torch.save(model.student.state_dict(), os.path.join(save_path, "%03d.pth" % (epoch+1)))
            
            logger.info("Epoch:{} model saved".format(epoch+1))
        
            
    end = time.time()
    
    logger.info("Training Finished total time={}".format(end-start_all))


def train_paraphraser(**kwargs):
   batch_size = kwargs["batch_size"]
   num_classes = kwargs["num_classes"]
   #learing_rate = kwargs["learing_rate"]
   epochs = kwargs["epochs"]
   teacher_pth = kwargs["pth"]
   types=kwargs["types"]
   dataset=kwargs["dataset"]
   teacher=kwargs["teacher"]
   student=kwargs["student"]
   assert types in ["shallow","depth"]
   if dataset == "GastricCancer":
        assert num_classes==2
   else:
        assert dataset=="Synapse"
        assert num_classes==9
   save_path="models/Paraphraser_"+types
   state_dict=torch.load(os.path.join("teacher/models/"+teacher_pth))
   if teacher=="FANet":
    net=UnetSKA2(in_ch=1,out_ch=num_classes)
   elif teacher=="Unet++":
    net=NestedUNet(in_ch=1,out_ch=num_classes)
   if student=="MobileNetV2_unet":
    
    model = Paraphraser(in_planes=256,planes=24)
   else:
    
    model = Paraphraser(in_planes=256,planes=64)
   
   if torch.cuda.device_count()>1:
    net=nn.DataParallel(net)
    model=nn.DataParallel(model)
    
   net.load_state_dict(state_dict)
   net.to(device)
   model.to(device)
   if dataset=="GastricCancer":
        custom_dataset = GastricCancerDataset(image_path="dataset/GastricCancer/train", transform=transforms.Compose([RandomHorizontalFlip(),
                                                                                               RandomVerticalFlip(),
                                                                                               ToTensor()]),type_image="resize")
        
   elif dataset=="Synapse": 
        custom_dataset = Synapsedataset(base_dir="dataset/Synapse",list_dir="dataset/Synapse/lists_Synapse",split="train",\
                                                        transform=transforms.Compose([RandomGenerator(output_size=[224,224])]))
   
   data_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)
    
   dt_size = len(data_loader.dataset)
   all_iter = math.ceil(dt_size/data_loader.batch_size)*epochs
   criterion=CriterionParaphraser()
   optimizer = optim.SGD(model.parameters(),
						lr = 0.1*0.1, 
						momentum = 0.9, 
						weight_decay = 2e-4)
   logger.info("Teacher == {}  Student == {}".format(teacher,student))
   logger_para.info("Start Training: Channel: {} Total epochs:{} Total iterations:{} Batch_Size:{} Training_Size:{} device:{}".format(num_classes,epochs,all_iter, batch_size, dt_size,device))
   all_loss=[]
   for epoch in range(epochs):
        logger_para.info("Epoch = {}".format(epoch+1))
        start = time.time()
        epoch_re_loss = 0
        step = 0
        for sample in data_loader:
            tmp_start = time.time()
            step += 1
            now_iter = step+epoch*math.ceil(dt_size/data_loader.batch_size)
            tmp_start=time.time()
            step += 1
            now_iter = step+epoch*((dt_size-1)//data_loader.batch_size+1)
            inputs = sample["image"].to(device)
            labels = sample["label"].to(device)
            labels = labels.long()
            adjust_learning_rate_poly(optimizer, all_iter, now_iter,1+epoch)
            optimizer.zero_grad()
            
            with torch.no_grad():
                middle_feature_shallow,middle_feature_depth,_=net(inputs)
            if(types=="shallow"):
                out=model(middle_feature_shallow.detach(),0)
                loss = criterion(out,middle_feature_shallow.detach())
            else:
                out=model(middle_feature_depth.detach(),0)
                loss = criterion(out,middle_feature_depth.detach())
            
            loss.backward()
            optimizer.step()
            epoch_re_loss+=loss.item()
            tmp_end = time.time()
        end = time.time()
        logger_para.info("第{}个Epoch:loss={:.5f},traing time={}".format(epoch+1, epoch_re_loss/step, end-start))
        all_loss.append(epoch_re_loss/step)
        if( (epoch + 1) % 5 == 0):
          if not os.path.exists(save_path):
              os.makedirs(save_path)
          torch.save(model.state_dict(), os.path.join(save_path, "%03d.pth" % (epoch+1)))
            
          logger_para.info("Epoch:{} model saved".format(epoch+1))
   end=time.time()
   epochs_list=[i for i in range(epochs)]
   plt.plot(epochs_list,all_loss,"or")
   plt.xlabel("epochs")
   plt.ylabel("training Reconstruct loss")
   plt.title("Paraphraser loss curve")
   plt.savefig("dataset/Reconstruct_loss.jpg")
    
   logger_para.info("Training Finished total time={}".format(end-start1))
   
   
    
def main():
    parse = argparse.ArgumentParser()
    parse.add_argument("-m", "--mode", type=str, default="train", help="running mode(default train)")
    parse.add_argument("-b", "--batch_size", type=int, default=2, help="batch_size")
    parse.add_argument("--student",type=str,default="MobileNetV2_unet")
    parse.add_argument("--teacher",type=str,default="FANet")
    parse.add_argument("-n", "--num_classes", type=int, default=2)
    parse.add_argument("--lr", type=float, default=3*1e-4)
    parse.add_argument("-e", "--epochs", type=int, default=10)
    parse.add_argument("-p", "--pth", default=False)
    parse.add_argument("-s","--lambda_s",type=float, default=1.0)
    parse.add_argument("-d","--lambda_d", type=float,default=1.0)
    
    parse.add_argument("-t","--types", default="csc")
    
    parse.add_argument("-a","--adversial",type=bool,default=False)
    parse.add_argument("--KD",type=bool,default=False)
    parse.add_argument("--adv_loss_type",type=str,default="wgan-gp")
    parse.add_argument("--lambda_adv",type=float,default=0.1)
    parse.add_argument("--dataset",type=str,default="GastricCancer")
    
    args = parse.parse_args()
    assert args.mode in ["train","train_paraphraser"]
    if args.mode == "train":
        train(
            batch_size=args.batch_size,
            num_classes=args.num_classes,
            learing_rate=args.lr,
            epochs=args.epochs,
            pth=args.pth,
            lambda_s=args.lambda_s,
            lambda_d=args.lambda_d,
            
            types=args.types,
            adversial=args.adversial,
            adv_loss_type=args.adv_loss_type,
            lambda_adv=args.lambda_adv,
            dataset=args.dataset,
            student=args.student,
            teacher=args.teacher,
            KD=args.KD
        )
        test_all(network=args.student, models="models/student",types=args.types,dataset=args.dataset,num_classes=args.num_classes)
    elif args.mode=="train_paraphraser":
        train_paraphraser(
            batch_size=args.batch_size,
            num_classes=args.num_classes,
            epochs=args.epochs,
            pth=args.pth,
            types=args.types,                
            dataset=args.dataset,
            student=args.student,
            teacher=args.teacher
        )
   
if __name__ == "__main__":
    main()

    
