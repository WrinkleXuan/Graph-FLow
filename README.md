# Graph Flow: Cross-layer Graph Flow Distillation for Dual Efficient Medical Image Segmentation


## Introduction
This repository contains the code  of our paper [Graph Flow: Cross-layer Graph Flow Distillation for Dual Efficient Medical Image Segmentation](https://arxiv.org/pdf/2203.08667.pdf).

The paper is under review in IEEE Transactions on Medical Imaging.

The paper is an extension version of our CoCo DistillNet which is published in the proceeding of the 2021 IEEE International Conference on
Bioinformatics and Biomedicine (BIBM).

We will collate our whole code as soon as possible.

## Visulatiztion on GastricCancer and Synpase
![kd_visualization_new](https://github.com/WrinkleXuan/Graph-FLow/blob/main/img/kd_visualization_new.pdf)

![visualization](https://github.com/WrinkleXuan/Graph-FLow/files/main/img/visualization.pdf)

## Visulatiztion of components ablation on GastricCancer and Synpase
![ablation_visualization](https://github.com/WrinkleXuan/Graph-FLow/blob/main/img/ablation_visualization.pdf)

## Visulatiztion on GastricCancer and Synpase with different student (Teahcer is TransUnet)
![student_visualization](https://github.com/WrinkleXuan/Graph-FLow/blob/main/img/student_visualization.pdf)


## Visulatiztion of semi-supervised learning on GastricCancer and Synpase
![semi_supervisied_visualization](https://github.com/WrinkleXuan/Graph-FLow/blob/main/img/semi_supervisied_visualization.pdf)

## Performance on GastricCancer and Synapse
<!--
### Network Efficiency 
<table class="tg">
<thead>
  <tr>
    <th class="tg-c3ow" colspan="2" rowspan="2">Networks</th>
    <th class="tg-c3ow" colspan="2">Gastric Cancer</th>
    <th class="tg-c3ow" colspan="2">Synapse</th>
    <th class="tg-0pky" rowspan="2">FLOPs(G)</th>
    <th class="tg-0pky" rowspan="2">Params(M)</th>
  </tr>
  <tr>
    <th class="tg-c3ow">ACC</th>
    <th class="tg-c3ow">mIOU</th>
    <th class="tg-c3ow">avergae DSC</th>
    <th class="tg-c3ow">avergage HD</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-c3ow" colspan="2">T: FANet</td>
    <td class="tg-c3ow">0.9030</td>
    <td class="tg-c3ow">0.8230</td>
    <td class="tg-c3ow">0.7953</td>
    <td class="tg-c3ow">25.409</td>
    <td class="tg-c3ow">171.556</td>
    <td class="tg-c3ow">38.250</td>
  </tr>
  <tr>
    <td class="tg-c3ow" rowspan="2"><br>Mobile U-Net </td>
    <td class="tg-0pky">w/o Grpah Flow</td>
    <td class="tg-c3ow">0.8555</td>
    <td class="tg-c3ow">0.7476</td>
    <td class="tg-c3ow">0.7382</td>
    <td class="tg-c3ow">34.327</td>
    <td class="tg-c3ow" rowspan="2"><br>1.492</td>
    <td class="tg-c3ow" rowspan="2"><br>4.640</td>
  </tr>
  <tr>
    <td class="tg-0pky">w/ Grpah Flow</td>
    <td class="tg-c3ow">0.8872</td>
    <td class="tg-c3ow">0.7973</td>
    <td class="tg-c3ow">0.78690</td>
    <td class="tg-c3ow">28.2401</td>
  </tr>
  <tr>
    <td class="tg-c3ow" rowspan="2"><br>ENet</td>
    <td class="tg-0pky">w/o Grpah Flow</td>
    <td class="tg-c3ow">0.8691</td>
    <td class="tg-c3ow">0.7684</td>
    <td class="tg-c3ow">0.7478</td>
    <td class="tg-c3ow">27.1688</td>
    <td class="tg-c3ow" rowspan="2"><br>0.516</td>
    <td class="tg-c3ow" rowspan="2"><br>0.349</td>
  </tr>
  <tr>
    <td class="tg-0pky">w/ Grpah Flow</td>
    <td class="tg-c3ow">0.8851</td>
    <td class="tg-c3ow">0.7936</td>
    <td class="tg-c3ow">0.7649</td>
    <td class="tg-c3ow">22.9843</td>
  </tr>
  <tr>
    <td class="tg-c3ow" rowspan="2"><br>ERFNet</td>
    <td class="tg-c3ow">w/o Grpah Flow</td>
    <td class="tg-c3ow">0.8695</td>
    <td class="tg-c3ow">0.7691</td>
    <td class="tg-c3ow">0.7501</td>
    <td class="tg-c3ow">31.5383</td>
    <td class="tg-c3ow" rowspan="2"><br>3.681<br></td>
    <td class="tg-c3ow" rowspan="2"><br>2.063</td>
  </tr>
  <tr>
    <td class="tg-0pky">w/ Grpah Flow</td>
    <td class="tg-c3ow">0.8889</td>
    <td class="tg-c3ow">0.8000</td>
    <td class="tg-c3ow">0.7674</td>
    <td class="tg-c3ow">27.7631</td>
  </tr>
</tbody>
</table>
-->
<!--
<table class="tg">
<thead>
  <tr>
    <th class="tg-0lax" colspan="2" rowspan="2">Knowledge <br><br>Distillation <br> </th>
    <th class="tg-baqh" colspan="2">Gastric Cancer</th>
    <th class="tg-baqh" colspan="10">Synapse</th>
  </tr>
  <tr>
    <th class="tg-0lax">ACC</th>
    <th class="tg-0lax">mIOU</th>
    <th class="tg-0lax">average DSC</th>
    <th class="tg-0lax">average HD</th>
    <th class="tg-0lax">aorta</th>
    <th class="tg-0lax">gallbladder</th>
    <th class="tg-0lax">left kidney</th>
    <th class="tg-0lax">right kidney</th>
    <th class="tg-0lax">liver</th>
    <th class="tg-0lax">pancreas</th>
    <th class="tg-0lax">spleen</th>
    <th class="tg-0lax">stomach</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-baqh">T:FANet</td>
    <td class="tg-0lax" rowspan="2"><br>Years</td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax">0.7950&lt;sup&gt;0.003</td>
    <td class="tg-0lax">24.5175</td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
  </tr>
  <tr>
    <td class="tg-0lax">S:Mobile U-Net</td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax">0.7331&lt;sup&gt;0.003</td>
    <td class="tg-0lax">40.7060</td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
  </tr>
  <tr>
    <td class="tg-0lax">KD</td>
    <td class="tg-0lax">2015</td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax">0.7719</td>
    <td class="tg-0lax">34.8758</td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
  </tr>
  <tr>
    <td class="tg-0lax">AT</td>
    <td class="tg-0lax">2016</td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax">0.7449</td>
    <td class="tg-0lax">37.9143</td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
  </tr>
  <tr>
    <td class="tg-0lax">FSP</td>
    <td class="tg-0lax">2017</td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax">0.7619</td>
    <td class="tg-0lax">33.2281</td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
  </tr>
  <tr>
    <td class="tg-0lax">FT</td>
    <td class="tg-0lax">2018</td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax">0.7790</td>
    <td class="tg-0lax">31.7290</td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
  </tr>
  <tr>
    <td class="tg-0lax">VID</td>
    <td class="tg-0lax">2019</td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax">0.7803</td>
    <td class="tg-0lax">33.8104</td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
  </tr>
  <tr>
    <td class="tg-0lax">SKD</td>
    <td class="tg-0lax">2020</td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax">0.7694</td>
    <td class="tg-0lax">37.0198</td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
  </tr>
  <tr>
    <td class="tg-0lax">IFVD</td>
    <td class="tg-0lax">2020</td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
  </tr>
  <tr>
    <td class="tg-0lax">EMKD</td>
    <td class="tg-0lax">2021</td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
  </tr>
  <tr>
    <td class="tg-0lax">CoCoD</td>
    <td class="tg-0lax">2021</td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
  </tr>
  <tr>
    <td class="tg-0lax">Graph Flow</td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
  </tr>
</tbody>
</table>
-->
### Annotation Efficiency 

![semi_supervised](https://github.com/WrinkleXuan/Graph-FLow/blob/main/img/semi_supervised.pdf)


## Models
<table class="tg">
<thead>
  <tr>
    <th class="tg-0pky"></th>
    <th class="tg-7btt">GastricCancer</th>
    <th class="tg-7btt">Synapse</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-7btt">Network Efficiency</td>
    <td class="tg-0pky"><a href="https://drive.google.com/drive/folders/1yOnINbK4BcVlO9n0KoaP39CJ2twCgwqq">Network Efficiency</a></td>
    <td class="tg-0pky"><a href="https://drive.google.com/drive/folders/1vthjYu1bVZwUa_6W0C5F0pjjKQvbyTEZ">Network Efficiency</a></td>
  </tr>
  <tr>
    <td class="tg-7btt">Annotation Efficiency</td>
    <td class="tg-0pky"><a href="https://drive.google.com/drive/folders/1mZUxvYrplhZapgq27XDQM6Fcg8-VYIP8">Annotation Efficiency</a></td>
    <td class="tg-0pky"><a href="https://drive.google.com/drive/folders/1yTVGOfp_j31QnKamtdk1E2jU8JXXvKwm">Annotation Efficiency</a></td>
  </tr>
</tbody>
</table>

## Supplementary Experiments

### The ablation study of different |L|s
<table class="tg">
<thead>
  <tr>
    <th class="tg-baqh" colspan="2" rowspan="2">&nbsp;&nbsp;&nbsp;<br>|L|<br><br> </th>
    <th class="tg-0lax" colspan="2">Gastric Cancer </th>
    <th class="tg-baqh" colspan="2">Synapse</th>
  </tr>
  <tr>
    <th class="tg-0lax">ACC</th>
    <th class="tg-0lax">mIOU</th>
    <th class="tg-0lax">average DSC</th>
    <th class="tg-0lax">average HD</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-baqh" colspan="2">|L|=1</td>
    <td class="tg-0lax">0.8872</td>
    <td class="tg-0lax">0.7973</td>
    <td class="tg-baqh">0.7874</td>
    <td class="tg-baqh">29.4551</td>
  </tr>
  <tr>
    <td class="tg-baqh" colspan="2">|L|=2</td>
    <td class="tg-0lax">0.8874</td>
    <td class="tg-0lax">0.7974</td>
    <td class="tg-baqh">0.7875</td>
    <td class="tg-amwm">28.7406</td>
  </tr>
  <tr>
    <td class="tg-baqh" colspan="2">|L|=3</td>
    <td class="tg-1wig">0.8877</td>
    <td class="tg-1wig">0.7980</td>
    <td class="tg-amwm">0.7886</td>
    <td class="tg-baqh">29.4536</td>
  </tr>
</tbody>
</table>

### The ablation study of hyperparameters

<table class="tg">
<thead>
  <tr>
    <th class="tg-baqh">Teacher: FANet</th>
    <th class="tg-baqh" colspan="4">Hyperparameters</th>
    <th class="tg-baqh" colspan="2">Gastric Cancer</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-baqh" rowspan="8"><br><br><br><br>Student: Mobile U-Net<br><br><br><br></td>
    <td class="tg-baqh">&lambda;<sub>1</td>
    <td class="tg-baqh">&lambda;<sub>2</td>
    <td class="tg-baqh">&lambda;<sub>3</td>
    <td class="tg-baqh">&lambda;<sub>4</td>
    <td class="tg-baqh">ACC</td>
    <td class="tg-baqh">mIOU</td>
  </tr>
  <tr>
    <td class="tg-baqh">1</td>
    <td class="tg-baqh">1</td>
    <td class="tg-baqh">1</td>
    <td class="tg-baqh">1</td>
    <td class="tg-baqh">0.7147</td>
    <td class="tg-baqh">0.5560</td>
  </tr>
  <tr>
    <td class="tg-baqh">1</td>
    <td class="tg-baqh">10<sup>-4</td>
    <td class="tg-baqh">1</td>
    <td class="tg-baqh">1</td>
    <td class="tg-baqh">0.7151</td>
    <td class="tg-baqh">0.5565</td>
  </tr>
  <tr>
    <td class="tg-baqh">1</td>
    <td class="tg-baqh">10<sup>-9</td>
    <td class="tg-baqh">1</td>
    <td class="tg-baqh">1</td>
    <td class="tg-baqh">0.7081</td>
    <td class="tg-baqh">0.5481</td>
  </tr>
  <tr>
    <td class="tg-baqh">10<sup>-3</td>
    <td class="tg-baqh">10<sup>-9</td>
    <td class="tg-baqh">1</td>
    <td class="tg-baqh">1</td>
    <td class="tg-baqh">0.8781</td>
    <td class="tg-baqh">0.7827</td>
  </tr>
  <tr>
    <td class="tg-baqh">10<sup>-5</td>
    <td class="tg-baqh">10<sup>-4</td>
    <td class="tg-baqh">1</td>
    <td class="tg-baqh">1</td>
    <td class="tg-baqh">0.7230</td>
    <td class="tg-baqh">0.5661</td>
  </tr>
  <tr>
    <td class="tg-baqh">10<sup>-5</td>
    <td class="tg-baqh">10<sup>-9</td>
    <td class="tg-baqh">1</td>
    <td class="tg-baqh">1</td>
    <td class="tg-baqh">0.8800</td>
    <td class="tg-baqh">0.7857</td>
  </tr>
  <tr>
    <td class="tg-baqh">10<sup>-5</td>
    <td class="tg-baqh">10<sup>-9</td>
    <td class="tg-baqh">0.1</td>
    <td class="tg-baqh">1</td>
    <td class="tg-amwm">0.8874</td>
    <td class="tg-amwm">0.7974</td>
  </tr>
</tbody>
</table>

## Requirments

* Python 3.6
* Pytorch 1.7.1
* Two NVIDIA TITAN XP GPUs


## Acknowledgement
The codebase of semantic segmentation is succeed from the work of my senior schoolmates [Accurate Retinal Vessel Segmentation in Color Fundus Images via Fully Attention-Based Networks](https://ieeexplore.ieee.org/abstract/document/9210783). 

The codebase of kd is heavily borrowed from [Knowledge-Distillation-Zoo](https://github.com/AberHu/Knowledge-Distillation-Zoo) and [structure_knowledge_distillation](https://github.com/irfanICMLL/structure_knowledge_distillation) .

The pre-processed Synapse is from [Transunet](https://github.com/Beckschen/TransUNet).

Thanks for their excellent works.
