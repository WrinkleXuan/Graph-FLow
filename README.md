# Graph Flow: Cross-layer Graph Flow Distillation for Dual Efficient Medical Image Segmentation


## Introduction
This repository contains the code  of our paper Graph Flow: Cross-layer Graph Flow Distillation for Dual Efficient Medical Image Segmentation.

The paper is under review in IEEE Transactions on Medical Imaging.

The paper is an extension version of our CoCo DistillNet which is published in the proceeding of the 2021 IEEE International Conference on
Bioinformatics and Biomedicine (BIBM).

We will collate our whole code as soon as possible.

## Visulatiztion on GastricCancer and Synpase
![kd_visualization_new](https://user-images.githubusercontent.com/84963829/160541226-c0ea02c5-5995-4116-906d-3e756e82156a.png)

## Performance on GastricCancer and Synapse 
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

## Requirments

* Python 3.6
* Pytorch 1.7.1
* Two NVIDIA TITAN XP GPUs


## Acknowledgement
The codebase of semantic segmentation is succeed from the work of my senior schoolmates [Accurate Retinal Vessel Segmentation in Color Fundus Images via Fully Attention-Based Networks](https://ieeexplore.ieee.org/abstract/document/9210783). 

The codebase of kd is heavily borrowed from [Knowledge-Distillation-Zoo](https://github.com/AberHu/Knowledge-Distillation-Zoo) and [structure_knowledge_distillation](https://github.com/irfanICMLL/structure_knowledge_distillation) .

The pre-processed Synapse is from [Transunet](https://github.com/Beckschen/TransUNet).

Thanks for their excellent works.
