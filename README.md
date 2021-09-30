# Anomaly Detection on Graphs

Implementation of the PANDA anomaly detector for graphs, using deep learning.
[“PANDA: Adapting Pretrained Features for Anomaly Detection and Segmentation” (CVPR 2021)](https://arxiv.org/pdf/2010.05903.pdf)

The PANDA method for graphs is based on a pretrained feature extractor 
which is then trained to find anomalies using the compactness loss. 

In this project, as the pretrained model (for features extraction), I used a basic [dgl](https://www.dgl.ai/) model, 
and the models suggested in [Self-supervised Learning on Graphs:
Deep Insights and New Directions](https://arxiv.org/pdf/2006.10141.pdf).

**Github references**

>PANDA github: [https://github.com/talreiss/PANDA](https://github.com/talreiss/PANDA)

>SSL pretrained tasks github: [https://github.com/ChandlerBang/SelfTask-GNN](https://github.com/ChandlerBang/SelfTask-GNN)


**Experiments**
---


**Requirements**

You need to install ica package, for running the SSL tasks yourself:
```
pip uninstall ica # in case you have installed it before
git clone https://github.com/ChandlerBang/ica.git 
cd ica
python setup.py install
```
**Code Arguments** 
```
> Each run takes one ssl task and graph dataset and runs the PANDA method on both.

Available datasets: cora, pubmed, citeseer
Available tasks: PairwiseDistance, EdgeMask, PairwiseAttrSim, Distance2Labeled, AttributeMask, NodeProperty
You can choose the task 'best', which then chooses the best task for the given dataset according to the paper findings

> The file args.csv should be changed according to the specific arguments you wish to run.
```
**Run the code**
```
git clone https://github.com/YasminHeimann/Anomaly-Detection-on-Graphs
cd Graph_Anomaly
pip install -r requirements.txt
```
