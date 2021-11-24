# PCAN for Multiple Object Tracking and Segmentation

This is the offical implementation of paper [PCAN](https://arxiv.org/abs/2106.11958) for MOTS. 

We also present a [trailer](https://www.youtube.com/watch?v=Vt-kYDPfh10) that consists of method illustrations and tracking & segmentation visualizations. Our project website contains more information: [vis.xyz/pub/pcan](https://www.vis.xyz/pub/pcan/). Code is under organization.

> [**Prototypical Cross-Attention Networks for Multiple Object Tracking and Segmentation**](https://papers.nips.cc/paper/2021/file/093f65e080a295f8076b1c5722a46aa2-Paper.pdf)  
> **NeurIPS 2021, Spotlight**  
> Lei Ke, Xia Li, Martin Danelljan, Yu-Wing Tai, Chi-Keung Tang, Fisher Yu


<div align="center">
<img src="./figures/pcan_banner_new.gif" width="100%" />
</div>


## Abstract

Multiple object tracking and segmentation requires detecting, tracking, and segmenting objects belonging to a set of given classes. Most approaches only exploit the temporal dimension to address the association problem, while relying on single frame predictions for the segmentation mask itself. We propose Prototypical Cross-Attention Network (PCAN), capable of leveraging rich spatio-temporal information for online multiple object tracking and segmentation. PCAN first distills a space-time memory into a set of prototypes and then employs cross-attention to retrieve rich information from the past frames. To segment each object, PCAN adopts a prototypical appearance module to learn a set of contrastive foreground and background prototypes, which are then propagated over time. Extensive experiments demonstrate that PCAN outperforms current video instance tracking and segmentation competition winners on both Youtube-VIS and BDD100K datasets, and shows efficacy to both one-stage and two-stage segmentation frameworks. 

## Prototypical Cross-Attention Networks (PCAN)
<img src="figures/pcan-banner-final.png" width="800">

## Main results
#### Results on BDD100K

| Detector  | mMOTSA-val | mIDF1-val | ID Sw.-val |                                            Scores-val                                            | mMOTSA-test | mIDF1-test | ID Sw.-test |                                            Scores-test                                            |                                             Config                                             |                                                                                        Weights                                                                                         |                                           Preds                                           |                                            Visuals                                            |
| :-------: | :--------: | :-------: | :--------: | :----------------------------------------------------------------------------------------------: | :---------: | :--------: | :---------: | :-----------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------: |
| ResNet-50 |    28.1    |   45.4    |    874     | [scores](https://dl.cv.ethz.ch/bdd100k/mots/scores-val/pcan-frcnn_r50_fpn_12e_mots_bdd100k.json) |    31.9     |    50.4    |     845     | [scores](https://dl.cv.ethz.ch/bdd100k/mots/scores-test/pcan-frcnn_r50_fpn_12e_mots_bdd100k.json) | [config](https://github.com/SysCV/pcan/blob/main/configs/segtrack-frcnn_r50_fpn_12e_bdd10k.py) | [model](https://dl.cv.ethz.ch/bdd100k/mots/models/pcan-frcnn_r50_fpn_12e_mots_bdd100k.pth) \| [MD5](https://dl.cv.ethz.ch/bdd100k/mots/models/pcan-frcnn_r50_fpn_12e_mots_bdd100k.md5) | [preds](https://dl.cv.ethz.ch/bdd100k/mots/preds/pcan-frcnn_r50_fpn_12e_mots_bdd100k.zip) | [visuals](https://dl.cv.ethz.ch/bdd100k/mots/visuals/pcan-frcnn_r50_fpn_12e_mots_bdd100k.zip) |

## Installation
Please refer to [INSTALL.md](docs/INSTALL.md) for installation instructions.

## Usages
Please refer to [GET_STARTED.md](docs/GET_STARTED.md) for dataset preparation and running instructions.


## Citation
If you find PCAN useful in your research or refer to the provided baseline results, please star :star: this repository and consider citing :pencil::
```
@inproceedings{pcan,
    author    = {Ke, Lei and Li, Xia and Danelljan, Martin and Tai, Yu-Wing and Tang, Chi-Keung and Yu, Fisher},
    booktitle = {Advances in Neural Information Processing Systems},
    title     = {Prototypical Cross-Attention Networks for Multiple Object Tracking and Segmentation},
    year      = {2021}
}
