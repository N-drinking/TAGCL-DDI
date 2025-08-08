# TAGCL-DDI

## Run code
For how to use ***,
about environment
networkx                 3.0
numpy                    1.24.1
pandas                   2.0.3
rdkit                    2024.3.5
scikit-learn             1.3.2
torch                    2.2.1+cu118
torch_geometric          2.5.2
torch_scatter            2.1.2+pt22cu118
torch_sparse             0.6.18+pt22cu118
torchaudio               2.2.1+cu118
torchvision              0.17.1+cu118
tqdm                     4.66.2
wheel                    0.41.2


一、about the Deng's dataset.

1.Learning drug structural features from drug molecular graphs,
# 我们用到的文件是drug_listxiao.csv，需要在trimnet对应文件中更改路经
```
python drugfeature_fromMG.py
```
# 运行后将得到5个不同折数据的药物特征的npy文件，和一个关于药物id的npy文件
2.Training/validating/testing for 5 times and get the average scores of multiple metrics.
# Deng数据集中用到的药物作用类型有65类，我们需要将对应输入输出代码中的维度设置为65
```
python 5timesrun.py
```

3.You can see the final results of 5 runs in 'test.txt'

二、about the Ryu's dataset.
1.Learning drug structural features from drug molecular graphs, 
# 我们用到的文件是drug_smiles.csv，需要在trimnet对应文件中更改路经
```
python drugfeature_fromMG.py
```
# 运行后将得到5个不同折数据的药物特征的npy文件，和一个关于药物id的npy文件

2.Training/validating/testing for 5 times and get the average scores of multiple metrics.
# Ryu数据集中用到的药物作用类型有86类，我们需要将对应输入输出代码中的维度设置为86
```
python 5timesrun.py
```

3.You can see the final results of 5 runs in 'test.txt'

三、
# 运行以上代码中所有涉及到data文件夹数据的都要更换为相关数据集中给出的data文件夹