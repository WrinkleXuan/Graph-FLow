# Graph Flow: Cross-layer Graph Flow Distillation for Dual Efficient Medical Image Segmentation


## Introduction
This repository contains the code  of our paper [Graph Flow: Cross-layer Graph Flow Distillation for Dual Efficient Medical Image Segmentation](https://arxiv.org/pdf/2203.08667.pdf).

The paper is under review in IEEE Transactions on Medical Imaging.

The paper is an extension version of our CoCo DistillNet which is published in the proceeding of the 2021 IEEE International Conference on
Bioinformatics and Biomedicine (BIBM).

We will collate our whole code as soon as possible.

## Visulatiztion on GastricCancer and Synpase
[kd_visualization_new](https://user-images.githubusercontent.com/84963829/160541226-c0ea02c5-5995-4116-906d-3e756e82156a.png)
[visualization.pdf](https://github.com/WrinkleXuan/Graph-FLow/files/9309893/visualization.pdf)

## Visulatiztion of components ablation on GastricCancer and Synpase
[ablation_visualization.pdf](https://github.com/WrinkleXuan/Graph-FLow/files/9309878/ablation_visualization.pdf)

## Visulatiztion on GastricCancer and Synpase with different student (Teahcer is TransUnet)
[student_visualization.pdf](https://github.com/WrinkleXuan/Graph-FLow/files/9309896/student_visualization.pdf)


## Visulatiztion of semi-supervised learning on GastricCancer and Synpase
[semi_supervisied_visualization.pdf](https://github.com/WrinkleXuan/Graph-FLow/files/9309881/semi_supervisied_visualization.pdf)

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
### Annotation Efficiency 

<!--[semi_supervised](https://user-images.githubusercontent.com/84963829/160548664-fad9843a-a87d-4b09-9090-411a08bff7e9.png)-->

<!--
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
-->

## Requirments

* Python 3.6
* Pytorch 1.7.1
* Two NVIDIA TITAN XP GPUs


## Acknowledgement
The codebase of semantic segmentation is succeed from the work of my senior schoolmates [Accurate Retinal Vessel Segmentation in Color Fundus Images via Fully Attention-Based Networks](https://ieeexplore.ieee.org/abstract/document/9210783). 

The codebase of kd is heavily borrowed from [Knowledge-Distillation-Zoo](https://github.com/AberHu/Knowledge-Distillation-Zoo) and [structure_knowledge_distillation](https://github.com/irfanICMLL/structure_knowledge_distillation) .

The pre-processed Synapse is from [Transunet](https://github.com/Beckschen/TransUNet).

Thanks for their excellent works.
