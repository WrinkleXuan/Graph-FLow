"""
zwx
2021-4-27
"""


from PIL import Image
import numpy as np
import  os
from  glob import  glob
import argparse
import math
import torchvision.transforms as transforms 
def crop_picture_train(path,save_path,cols,rows,overlap):
    #Pad = transforms.Pad(padding=384,padding_mode="reflect")
    if not os.path.exists(save_path):
        os.makedirs(os.path.join(save_path, "image"))
        os.makedirs(os.path.join(save_path, "label"))
    images_list = glob(os.path.join(path, "*.jpg"))
    #labels_list = glob(os.path.join(path, "*.png"))
    for image_path in images_list:
        
        image = Image.open(image_path)
        #image = Pad(image)
        label_path = image_path.split(".")[0]+".png"
        label = Image.open(label_path)
        #label = Pad(label)
        w, h = image.size
        crop_rows = math.ceil(h/(rows-overlap))
        crop_cols = math.ceil(w/(cols-overlap))
        for i in range(crop_cols):
            for j in range(crop_rows):
                Rx = (i+1)*cols-i*overlap
                Ry = (j+1)*rows-j*overlap
                Lx = Rx - cols
                Ly = Ry - rows

                image_cropped = image.crop((Lx, Ly, Rx, Ry)) 
                image_cropped.save(os.path.join(save_path, "image/%s" % (image_path.split("/")[-1].split(".")[0] +"_"+str(j)+"_"+str(i)+".png")))
                
                label_cropped = label.crop((Lx, Ly, Rx, Ry))
                
                label_cropped.save(os.path.join(save_path, "label/%s" % (label_path.split("/")[-1].split(".")[0] +"_" + str(j) +"_"+str(i)+".png")))

def crop_picture_test(path,save_path,cols,rows,overlap): #对test进行镜像填充 
    Pad = transforms.Pad(padding=384,padding_mode="reflect") #当size为1024*1024时 预测时只保留中间512*512的部分 
    if not os.path.exists(save_path):
        os.makedirs(os.path.join(save_path, "image"))
        os.makedirs(os.path.join(save_path, "label"))
    images_list = glob(os.path.join(path, "*.jpg"))
    
    for image_path in images_list:
     
        image = Image.open(image_path)
        label_path = image_path.split(".")[0]+".png"
        
        label = Image.open(label_path)
        w, h = image.size  #原图大小
        image = Pad(image)
        label = Pad(label) 
        #w, h = image.size
        #crop_rows = math.ceil(h/(rows-overlap))
        #crop_cols = math.ceil(w/(rows-overlap)) 
        crop_rows = w // (rows-overlap)
        crop_cols = h // (cols-overlap)
        
        for i in range(crop_cols):
            for j in range(crop_rows):
                Rx = (i+1)*cols-i*overlap
                Ry = (j+1)*rows-j*overlap
                Lx = Rx - cols
                Ly = Ry - rows
                image_cropped = image.crop((Lx, Ly, Rx, Ry)) 
                image_cropped.save(os.path.join(save_path, "image/%s" % (image_path.split("/")[-1].split(".")[0] +"_"+str(j)+"_"+str(i)+".png")))
                label_cropped = label.crop((Lx, Ly, Rx, Ry))
                label_cropped.save(os.path.join(save_path, "label/%s" % (label_path.split("/")[-1].split(".")[0] +"_" + str(j) +"_"+str(i)+".png")))
                
                
                
def resize(path,save_path,size):
    if not os.path.exists(save_path):
        os.makedirs(os.path.join(save_path, "image"))
        os.makedirs(os.path.join(save_path, "label"))
    images_list = glob(os.path.join(path, "*.jpg"))
    #labels_list = glob(os.path.join(path, "*.png"))
    for image_path in images_list:
        #image_path = image_path.split("//")[0]+"/"+image_path.split("//")[-1]
        image = Image.open(image_path)
        label_path = image_path.split(".")[0]+".png"
        #os.path.join(image_path.split(".")[0],"_1stHO.png")
        label = Image.open(label_path)
        image_interpolation = Image.BILINEAR
        label_interpolation = Image.NEAREST
        image = image.resize((size,size),image_interpolation)
        label = label.resize((size,size),label_interpolation)
        image.save(os.path.join(save_path, "image/%s" % (image_path.split("/")[-1].split(".")[0]+".png")))
        label.save(os.path.join(save_path, "label/%s" % (label_path.split("/")[-1].split(".")[0]+".png")))
        
        
def resize_new(path,save_path,size):
    if not os.path.exists(save_path):
        os.makedirs(os.path.join(save_path, "image"))
        os.makedirs(os.path.join(save_path, "label"))
    images_list = glob(os.path.join(os.path.join(path,"image"), "*.png"))
    #labels_list = glob(os.path.join(path, "*.png"))
    for image_path in images_list:
        #image_path = image_path.split("//")[0]+"/"+image_path.split("//")[-1]
        image = Image.open(image_path)
        label_path = os.path.join(os.path.join(path,"label"),image_path.split("/")[-1])
        path.split(".")[0]+".png"
        #os.path.join(image_path.split(".")[0],"_1stHO.png")
        label = Image.open(label_path)
        image_interpolation = Image.BILINEAR
        label_interpolation = Image.NEAREST
        image = image.resize((size,size),image_interpolation)
        label = label.resize((size,size),label_interpolation)
        image.save(os.path.join(save_path, "image/%s" % (image_path.split("/")[-1].split(".")[0]+".png")))
        label.save(os.path.join(save_path, "label/%s" % (label_path.split("/")[-1].split(".")[0]+".png")))

def merge_picture(path, picturename,save_path,cols,rows,overlap):
    filename = glob(os.path.join(path, picturename+"*.png"))
    
    max_rows = 0
    max_cols = 0
    for name in filename:
        row = name.split("_")[-2]
        col = name.split("_")[-1].split(".")[0]
        if(int(row)>max_rows):
            max_rows = int(row)
        if(int(col)>max_cols):
            max_cols = int(cols)
    image = Image.open(filename[0])
    w, h = image.size
    #w=64
    #h=64
    num_rows = max_rows +1
    num_cols = max_rows +1

    dst = np.zeros((num_rows*h-(num_rows-1)*overlap, num_cols*w-(num_cols-1)*overlap))
    dst_count = np.zeros((num_rows*h-(num_rows-1)*overlap, num_cols*w-(num_cols-1)*overlap))
    
    for i in range(len(filename)):
        image = Image.open(filename[i])
        #image = image.crop((95,95,95+64,95+64))
        dst_i = np.zeros((num_rows * h-(num_rows-1)*overlap, num_cols * w-(num_cols-1)*overlap))
        dst_i_count = np.zeros((num_rows * h-(num_rows-1)*overlap, num_cols * w-(num_cols-1)*overlap))
        image = np.array(image)
        cols_th = int(filename[i].split("_")[-1].split(".")[0])
        rows_th = int(filename[i].split("_")[-2])
        Ry = (rows_th + 1) * h - rows_th * overlap
        Rx = (cols_th + 1) * w - cols_th * overlap

        Lx = Rx - w
        Ly = Ry - h
        #print(Ly, Ry, Lx,Rx)
        #print(dst_i[Ly:Ry, Lx:Rx].shape)

        dst_i[Ly:Ry, Lx:Rx] = image

        dst_i_count[Ly:Ry, Lx:Rx] = 1
        dst += dst_i
        dst_count += dst_i_count
    merge_pic = dst/dst_count
    """
    h,w=merge_pic.shape
    for i in range(h):
        for j in  range(w):
            if np.uint8(merge_pic)[i][j]!=0:
                print(merge_pic[i][j])
    """
    merge_image = Image.fromarray(np.uint8(merge_pic))
    
    merge_image = merge_image.crop((0, 0,cols,rows))
    merge_image.save(os.path.join(save_path, picturename+".png"))
    

if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument("-m", "--mode", type=str, default="merge", help="merge||patch||resize")
    parse.add_argument("-dir",type=str, default="train", help="train||test")
    
    args = parse.parse_args()
    assert args.mode in ["merge", "patch","resize"]
    if(args.mode == "merge"):
      test_list = glob("dataset/final_digital_pathology_dataset/test/*.png") 
      if not os.path.exists("dataset/merge"):
        os.mkdir("dataset/merge") 
      for t in test_list:
        test_name=t.split("/")[-1].split(".")[0]
        merge_picture("dataset/prediction", test_name, "dataset/merge", 512,512, 0)
      #merge_picture("dataset/resize/train/label", "A9C6BEA25BDB3100F175D7EE82C0AF6D", "dataset/merge", 512,512, 0)
       
    elif(args.mode == "patch"):
      crop_picture_train("dataset/final_digital_pathology_dataset/train", "dataset/train", 1024,1024, 0)
      crop_picture_test("dataset/final_digital_pathology_dataset/test", "dataset/test", 1024, 1024, 1024-256)
    else:
      resize_new("dataset/train","dataset/resize/train",256)
      resize_new("dataset/test","dataset/resize/test",256)
