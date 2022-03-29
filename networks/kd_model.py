import os
import logging
import torch
import  torch.nn as nn
import torch.optim as optim
from itertools import chain
from networks.Unet import  Unet
from networks.Unet_SKA2 import UnetSKA2
from networks.Unetpp import NestedUNet
from networks.MobileNetV2_unet import MobileNetV2_unet
from networks.enet import ENet
from networks.erfnet import ERFNet
from utils.criterion import CriterionKD,CriterionSpatial,CriterionChannel,CriterionSpatialwithChannel,AT,\
Cross_layer_CriterionSpatialwithChannel,CriterionGraph,CriterionAdv,CriterionAdvForG,CriterionAdditionalGP,GANLoss,CriterionIFV,CriterionPixelWise,CriterionPairWiseforWholeFeatAfterPool,FSP
from  networks.feature_projection import Paraphraser,Translator
from networks.sagan_models import Discriminator,PixelDiscriminator,NLayerDiscriminator

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger =  logging.getLogger("kd")
logger.setLevel(logging.INFO)
fh = logging.FileHandler("kd.log")
sh = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s:%(levelname)s:[%(filename)s]:%(message)s")
fh.setFormatter(formatter)
fh.setLevel(logging.INFO)
sh.setFormatter(formatter)
sh.setLevel(logging.INFO)
logger.addHandler(fh)
logger.addHandler(sh)

def load_S_model(pth,model):
    logger.info("Stduent Model loaded from student/models/{}".format(pth))
    if os.path.exists(os.path.join("student/models", pth)):
        state_dict = torch.load(os.path.join("./student/moels", pth))

        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.load_state_dict(state_dict)
    else:
        logger.warning("Student Model pth does not exists")

def load_T_model(pth, model):
    logger.info('Teacher Model loaded from  teacher/models/{}'.format(pth))
    if os.path.exists(os.path.join("teacher/models", pth)):
        state_dict = torch.load(os.path.join("teacher/models", pth))

        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.load_state_dict(state_dict)

    else:
        logger.warning("Teahcer Model pth does not exists")
def load_Paraphraser(pth,model_shallow,model_depth):
    logger.info('Paraphraser Model loaded from  models/Paraphraser_shallow/{}'.format(pth))
    if os.path.exists(os.path.join("models/Paraphraser_shallow", pth)):
        state_dict = torch.load(os.path.join("models/Paraphraser_shallow", pth))

        if torch.cuda.device_count() > 1:
            model_shallow = nn.DataParallel(model_shallow)
        model_shallow.load_state_dict(state_dict)

    else:
        logger.warning("Paraphraser Model pth does not exists")
    logger.info('Paraphraser Model loaded from  models/Paraphraser_depth/{}'.format(pth))
    if os.path.exists(os.path.join("models/Paraphraser_depth", pth)):
        state_dict = torch.load(os.path.join("models/Paraphraser_depth", pth))

        if torch.cuda.device_count() > 1:
            model_depth = nn.DataParallel(model_depth)
        model_depth.load_state_dict(state_dict)

    else:
        logger.warning("Paraphraser Model pth does not exists")
class NetModel():
    def __init__(self,teacher,student,lr,lr_d,preprocess_GAN_mode,num_classes,batch_size,imsize_for_adv,adv_conv_dim,adv_loss_type,adversial=False,KD=False,student_pth=False, teacher_pth="085.pth",lambda_s=1.0,lambda_adv=1.0,lambda_d=0.1,mode="sp"):
        #load scratch student network
        #student = Unet()
        student_1=student
        if student=="MobileNetV2_unet":
            student = MobileNetV2_unet(num_classes=num_classes,pre_trained=None)
        elif student=="ENet":
            student = ENet(num_classes)
        elif student=="ERFNet":
            student = ERFNet(num_classes)
        
        
        self.lambda_s=lambda_s
        self.lambda_d=lambda_d
        self.lambda_adv=lambda_adv
        
        
        self.mode=mode
        self.lr_d=lr_d
        self.adversial=adversial
        self.KD=KD
        self.adv_loss_type=adv_loss_type
        if student_pth:
            load_S_model(student_pth,model=student)
            student.to(device)
            self.student = student    
        else:
            
            if (torch.cuda.device_count()> 1):
                student = nn.DataParallel(student)
            student.to(device)
            self.student=student
        #load  teacher
        #teacher=Unet()
        if teacher=="FANet":
            teacher=UnetSKA2(in_ch=1,out_ch=num_classes)
        elif teacher == "Unet++":
            teacher=NestedUNet(in_ch=1, out_ch=num_classes)
        load_T_model(pth=teacher_pth, model=teacher)
        teacher.to(device)
        self.teacher = teacher
                
        #load Paraphraser
        #mobileUnet
        #self.Paraphraser_shallow=Paraphraser(in_planes=256,planes=24)
        #self.Paraphraser_depth=Paraphraser(in_planes=256,planes=24)
        
        if student_1=="MobileNetV2_unet":
            self.Paraphraser_shallow=Paraphraser(in_planes=256,planes=24)
            self.Paraphraser_depth=Paraphraser(in_planes=256,planes=24)
            #self.Translator_shallow=Translator(in_planes=24,planes=num_classes)
            #self.Translator_depth=Translator(in_planes=24,planes=num_classes)
        else:
            #enet erfnet
            self.Paraphraser_shallow=Paraphraser(in_planes=256,planes=64)
            self.Paraphraser_depth=Paraphraser(in_planes=256,planes=64)
            #self.Translator_shallow=Translator(in_planes=64,planes=num_classes)
            #self.Translator_depth=Translator(in_planes=64,planes=num_classes)
        """
        if torch.cuda.device_count() > 1:
            self.Translator_shallow = nn.DataParallel(self.Translator_shallow)
            self.Translator_depth = nn.DataParallel(self.Translator_depth)
            self.Translator_shallow.to(device)
            self.Translator_depth.to(device)
        """    
        load_Paraphraser("195.pth",self.Paraphraser_shallow,self.Paraphraser_depth)  
        self.Paraphraser_shallow.to(device)
        self.Paraphraser_depth.to(device)
        
        #load Discriminator
        if(self.adversial==True):
            self.D_model=Discriminator(preprocess_GAN_mode,num_classes,batch_size,imsize_for_adv,adv_conv_dim)
            #self.D_model=PixelDiscriminator(num_classes+1)
            #self.D_model=NLayerDiscriminator(num_classes+1,n_layers=3)
            if(torch.cuda.device_count()>1):
                self.D_model=nn.DataParallel(self.D_model)
                self.D_model.to(device)

        
        self.optimizer = optim.Adam(self.student.parameters(), lr=lr, weight_decay=2e-4)
        if(self.adversial==True):
            self.D_solver = optim.Adam(self.D_model.parameters(), lr_d, [0.9, 0.99])
       
        
        self.criterion_ce = nn.CrossEntropyLoss()
        self.criterion_kd = CriterionKD().cuda()
        """
        self.criterion_sp = CriterionSpatial().cuda()
        self.criterion_ch = CriterionChannel().cuda()
        self.criterion_sc = CriterionSpatialwithChannel().cuda()
        """
        self.criterion_csc = Cross_layer_CriterionSpatialwithChannel().cuda()
        
        self.criterion_graph=CriterionGraph().cuda()
        #self.criterion_L1 = nn.L1Loss()
        if mode =="IFVD":
            self.criterion_IFVD=CriterionIFV(num_classes).cuda()
        if mode =="SKD":
            self.criterion_pixel_wise = CriterionPixelWise().cuda()
            self.criterion_pair_wise_for_interfeat=CriterionPairWiseforWholeFeatAfterPool(0.5, feat_ind=-5).cuda()
        if mode =="AT":
            self.criterion_at = AT(p=1).cuda()
        if mode =="FSP":
            self.criterion_fsp = FSP().cuda()
        
        if(self.adversial==True):
            self.criterionGAN =GANLoss().cuda()
            self.criterion_adv=CriterionAdv(self.adv_loss_type).cuda()
            self.criterion_adv_for_G=CriterionAdvForG(self.adv_loss_type).cuda()
            if(self.adv_loss_type=="wgan-gp"):
                self.criterion_AdditionalGP = CriterionAdditionalGP(self.D_model, 10.0).cuda()
        


    
    def adjust_learning_rate_poly(self,optimizer, all_iter, now_iter, epoch):
        
        # gastriccancer
        if  epoch<=50:
            #base_lr = 3*0.0003
            base_lr = 0.003
        
        elif epoch>50 and epoch<=100:
            #base_lr = 3*0.00003
            base_lr=0.0003
        elif epoch>100 and epoch<=150:
            #base_lr=3*0.00003
            base_lr = 0.00003
        elif epoch>150:
            #base_lr=3*0.00003
            base_lr = 0.000003
        
        
        """
        if epoch <= 100:
            #base_lr = 3*0.0003
            base_lr = 0.005
        
        elif epoch>100 and epoch<=200:
            #base_lr = 3*0.00003
            base_lr=0.005
        elif epoch>200:
            #base_lr=3*0.00003
            base_lr = 0.0003
        """
        #base_lr=0.0009 #csc synapse 
        #base_lr=0.0003 #graph synapse
        #base_lr=0.0000001
        lr = base_lr*((1-(now_iter/all_iter))**0.9)
        #lr = base_lr
        
        if(len(optimizer.param_groups)==1):
            optimizer.param_groups[0]["lr"] = lr
        else:
            
            optimizer.param_groups[0]['lr'] = lr*0.1
        
        """
        for i in range(0,len(optimizer.param_groups)):
            optimizer.param_groups[i]['lr'] = lr
        """
    def adjust_learning_rate(sefl,optimizer,base_lr,all_iter,now_iter,epoch):
        """
        if  epoch<=50:
            #base_lr = 3*0.0003
            base_lr = 0.0003
        elif epoch>50 and epoch<=100:
            #base_lr = 3*0.00003
            base_lr=0.00003
        elif epoch>100 and epoch<=150:
            #base_lr=3*0.00003
            base_lr = 0.000003
        elif epoch>150:
            #base_lr=3*0.00003
            base_lr = 0.0000003
        """
        """
        if epoch <= 50:
            #base_lr = 3*0.0003
            base_lr = 0.0003
        
        elif epoch>50 and epoch<=100:
            #base_lr = 3*0.00003
            base_lr=0.0003
        elif epoch>100 :
            #base_lr=3*0.00003
            base_lr = 0.000003
        """
        base_lr = 0.00003
        #lr = base_lr
        lr = base_lr*((1-(now_iter/all_iter))**0.9)
        optimizer.param_groups[0]['lr'] = lr
    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
    def set_input(self, data):
        self.image = data["image"].to(device)
        self.label = data["label"].to(device)
        self.label = self.label.long()

    def segmentation_forward(self):
        if self.mode!="scratch":
            with torch.no_grad():
                self.middel_feature_T_1,self.middel_feature_T,self.preds_T = self.teacher.eval()(self.image)
                if self.mode=="sc" or self.mode=="sp" or self.mode=="FSP" or self.mode=="Graph":
                 
                #if self.mode== "csc" or self.mode=="sc" or self.mode=="sp" or self.mode=="FSP" or self.mode=="Graph":
                    #print("loaded")
                    self.middel_feature_T_1=self.Paraphraser_shallow(self.middel_feature_T_1,1)
                    self.middel_feature_T=self.Paraphraser_depth(self.middel_feature_T,1)
                    


                
                
        self.middel_feature_S_1,self.middel_feature_S,self.preds_S = self.student.train()(self.image)
        
        
    def segmentation_backward(self, lambda_kd=1.0,lambda_s=1.0,lambda_adv=0.1):
        loss=0
        temp = self.criterion_ce(self.preds_S, self.label)
        self.ce_loss = temp.item()
        loss = temp
        
        if  self.mode=="csc":
            temp = lambda_s*self.criterion_csc(self.middel_feature_S_1,self.middel_feature_S,self.middel_feature_T_1,self.middel_feature_T)
            self.csc_loss = temp.item()
            loss = loss+temp
         
        elif self.mode=="Graph":

            
            temp_edges,temp_vertices=self.criterion_graph(self.middel_feature_S_1,self.middel_feature_S,self.middel_feature_T_1,self.middel_feature_T)
            #patch_size=3 GastricCancer
            #temp_edges=0.0000001*lambda_s*temp_edges
            #temp_edges=0.0000000001*lambda_s*temp_edges
            temp_edges=0.00000001*lambda_s*temp_edges
           

            #patch_size=3 GastricCancer
            #temp_vertices=0.0001*lambda_s*temp_vertices
            #temp_vertices=0.00001*lambda_s*temp_vertices
            temp_vertices=0.00001*lambda_s*temp_vertices
            
            
            
            self.graph_edges_loss=temp_edges.item()
            self.graph_vertices_loss=temp_vertices.item()
            loss=loss+temp_edges+temp_vertices
        elif self.mode=="IFVD":
            temp=lambda_s*self.criterion_IFVD(self.middel_feature_S,self.middel_feature_T,self.label)
            self.IFVD_loss = temp.item()  
            loss = loss+temp
        elif self.mode=="SKD":
            temp_1=lambda_s*self.criterion_pixel_wise(self.preds_S,self.preds_T)
            temp_2=0.1*lambda_s*self.criterion_pair_wise_for_interfeat(self.middel_feature_S_1,self.middel_feature_T_1)
           
            
            self.pi_loss=temp_1.item()
            self.pair_loss=temp_2.item()
            loss=loss+temp_1+temp_2
        elif self.mode=="AT":
            temp=lambda_s*self.criterion_at(self.middel_feature_S_1,self.middel_feature_T_1)
            temp+=lambda_s*self.criterion_at(self.middel_feature_S,self.middel_feature_T)
            
            self.at_loss=temp.item()
            loss=loss+temp
        elif self.mode=="FSP":
            temp=lambda_s*self.criterion_fsp(self.middel_feature_S_1,self.middel_feature_S,self.middel_feature_T_1,self.middel_feature_T)
           
            
            self.fsp_loss=temp.item()
            loss=loss+temp    

        if self.KD==True:
            temp = lambda_kd*self.criterion_kd(self.preds_S, self.preds_T)
            self.kd_loss = temp.item()
            loss = loss+temp
        
        if(self.adversial==True):
            #fake_AB = torch.cat((self.image, self.preds_S), 1)
            
            #pred_fake = self.D_model(fake_AB)
            
            pred_fake = self.D_model(self.preds_S)
            
            temp=lambda_adv*self.criterion_adv_for_G(pred_fake)
            self.adv_G_loss=temp.item()
            
            loss=loss+temp
            

            
        loss.backward()
        
        
        self.loss = loss.item()
    def discriminator_backword(self,lambda_d=1.0):
        """
        # Fake; stop backprop to the generator by detaching fake_B fake_B是student的输出
        fake_AB = torch.cat((self.image, self.preds_S), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.D_model(fake_AB.detach())
        real_AB = torch.cat((self.image, self.preds_T), 1)
        pred_real = self.D_model(real_AB)
 
       
        d_loss=self.criterion_adv(pred_fake,pred_real)
        self.w_distance=-d_loss.item()
        d_loss=lambda_d*d_loss
  
        if(self.adv_loss_type=="wgan-gp"): d_loss += lambda_d*self.criterion_AdditionalGP(fake_AB, real_AB)
        
        d_loss.backward()
        
        
        """
        pred_fake = self.D_model(self.preds_S.detach())
       
        pred_real = self.D_model(self.preds_T)
 
       
        d_loss=self.criterion_adv(pred_fake,pred_real)
        self.w_distance=-d_loss.item()
        d_loss=lambda_d*d_loss
  
        if(self.adv_loss_type=="wgan-gp"): d_loss += lambda_d*self.criterion_AdditionalGP(self.preds_S, self.preds_T)
        
        d_loss.backward()
        
    def optimize_parameters(self, data, lambda_kd, all_iter, now_iter, epoch):
        self.set_input(data)
        self.segmentation_forward()
        
        self.adjust_learning_rate_poly(self.optimizer,all_iter,now_iter,epoch)
        if(self.adversial==True):
            self.set_requires_grad(self.D_model,False)
        self.optimizer.zero_grad()
        self.segmentation_backward(lambda_kd=1,lambda_s=self.lambda_s,lambda_adv=self.lambda_adv)
        self.optimizer.step()
        
        if(True):
            if(self.adversial==True):
                self.set_requires_grad(self.D_model,True)
                self.adjust_learning_rate(self.D_solver,self.lr_d,all_iter,now_iter,epoch)
                self.D_solver.zero_grad()
                self.discriminator_backword(lambda_d=self.lambda_d)
                self.D_solver.step()
           







