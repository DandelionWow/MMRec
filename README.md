# $\text{MMRec}$

<div align="center">
  <a href="https://github.com/enoche/MultimodalRecSys"><img width="300px" height="auto" src="https://github.com/enoche/MMRec/blob/master/images/logo.png"></a>
</div>


$\text{MMRec}$: A modern <ins>M</ins>ulti<ins>M</ins>odal <ins>Rec</ins>ommendation toolbox that simplifies your research [arXiv](https://arxiv.org/abs/2302.03497).  
:point_right: Check our [comprehensive survey on MMRec, arXiv](https://arxiv.org/abs/2302.04473).   
:point_right: Check the awesome [multimodal recommendation resources](https://github.com/enoche/MultimodalRecSys).  

## Toolbox
<p>
<img src="./images/MMRec.png" width="500">
</p>

#### Please cite our paper if this framework helps you:
```
@article{zhou2023comprehensive,
      title={A Comprehensive Survey on Multimodal Recommender Systems: Taxonomy, Evaluation, and Future Directions}, 
      author={Hongyu Zhou and Xin Zhou and Zhiwei Zeng and Lingzi Zhang and Zhiqi Shen},
      year={2023},
      journal={arXiv preprint arXiv:2302.04473},
}

@article{zhou2023mmrecsm,
  author = {Zhou, Xin},
  title = {MMRec: Simplifying Multimodal Recommendation},
  year = {2023},
  journal={arXiv preprint arXiv:2302.03497},
}
```

## Supported Models
source code at: `src\models`

| **Model**       | **Paper (PDF)**                                                                                             | **Conference/Journal** | **Code**    |
|------------------|--------------------------------------------------------------------------------------------------------|------------------------|-------------|
| **General models**  |                                                                                                        |                        |             |
| SelfCF              | [SelfCF: A Simple Framework for Self-supervised Collaborative Filtering](https://arxiv.org/pdf/2107.03019.pdf)                                 | ACM TORS'23                  | selfcfed_lgn.py  |
| LayerGCN            | [Layer-refined Graph Convolutional Networks for Recommendation](https://arxiv.org/pdf/2207.11088.pdf)                                          | ICDE'23                | layergcn.py  |
| **Multimodal models**  |                                                                                                        |                        |             |
| VBPR              | [VBPR: Visual Bayesian Personalized Ranking from Implicit Feedback](https://arxiv.org/pdf/1510.01784.pdf)                                              | AAA'16                 | vbpr.py      |
| MMGCN             | [MMGCN: Multi-modal Graph Convolution Network for Personalized Recommendation of Micro-video](https://staff.ustc.edu.cn/~hexn/papers/mm19-MMGCN.pdf)               | MM'19              | mmgcn.py  |
| ItemKNNCBF             | [Are We Really Making Much Progress? A Worrying Analysis of Recent Neural Recommendation Approaches](https://arxiv.org/pdf/1907.06902.pdf)               | RecSys'19              | itemknncbf.py  |
| GRCN              | [Graph-Refined Convolutional Network for Multimedia Recommendation with Implicit Feedback](https://arxiv.org/pdf/2111.02036.pdf)            | MM'20      | grcn.py    |
| MVGAE             | [Multi-Modal Variational Graph Auto-Encoder for Recommendation Systems](https://ieeexplore.ieee.org/abstract/document/9535249)              | TMM'21     | mvgae.py   |
| DualGNN           | [DualGNN: Dual Graph Neural Network for Multimedia Recommendation](https://ieeexplore.ieee.org/abstract/document/9662655)                   | TMM'21     | dualgnn.py   |
| LATTICE           | [Mining Latent Structures for Multimedia Recommendation](https://arxiv.org/pdf/2104.09036.pdf)                                               | MM'21               | lattice.py  |
| SLMRec            | [Self-supervised Learning for Multimedia Recommendation](https://ieeexplore.ieee.org/document/9811387) |  TMM'22         |                  slmrec.py |
| **Newly added**  |                                                                                                        |                        |             |
| BM3         | [Bootstrap Latent Representations for Multi-modal Recommendation](https://arxiv.org/pdf/2207.05969.pdf)                                          | WWW'23                 | bm3.py |
| FREEDOM | [A Tale of Two Graphs: Freezing and Denoising Graph Structures for Multimodal Recommendation](https://arxiv.org/pdf/2211.06924.pdf)                                 | arxiv                  | freedom.py  |
| DRAGON  | [Enhancing Dyadic Relations with Homogeneous Graphs for Multimodal Recommendation](https://arxiv.org/pdf/2301.12097.pdf)                                 | arxiv                  | dragon.py  |

## ENV Init
```bash
conda env create -f ./env.yml
conda activate MMRec_ENV
```