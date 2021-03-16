# Machine-Learning-Experiment
Records about basic Machine Learning and four experiments
# 机器学习基本课程实验记录

## 介绍
机器学习与模式识别课程学习笔记记录与四次实验的报告记录

### 第一次实验报告
 **实验一** ：生成手写数字的特征向量数据集并存盘

 **处理步骤** 

1.确定数据区，这里固定为8*8
    
2.将数字区域平均分为3*3的小区域；
    
3.计算3*3的每个小区域中灰度平均值，第一行的6个比例保存到特征的前6个，第二行对应着特征的7~12个，以此类推。
    
4.可选：对36个特征做行投影变换，压缩为6个特征。
    
报告内容应包括：题目，实验原理，程序，实验结果及分析

 **实验原理** 

1、手写数字数据集

该数据集共有1979个样本，每个样本的大小为1*64，64个值记录的是灰度值，通过reshape函数将其转变为8*8矩阵，调用matplotlib包中的函数可以利用灰度值显示手写数字图像，其效果如下图所示。

![输入图片说明](https://images.gitee.com/uploads/images/2021/0316/161507_efa7edf7_8794964.png "屏幕截图.png")

2、数据集处理

将8*8矩阵以3*3小矩阵进行特征提取，每个相邻的3*3小矩阵相距一行或一列，共有36个小矩阵，特征值取9个像素灰度值的均值。这36个小矩阵的灰度均值以1*6矩阵存储在一个元组中。
根据参考代码完善主要步骤的原理

 **程序** 

```
#导入数据处理包和数据集
import numpy as np
from sklearn.datasets import load_digits

#计算特征矩阵的函数定义,将矩阵平分为3*3的小区域，计算小区域的灰度平均值
def feature(img):
    new_img1 = np.zeros((6,6))
    new_img2 = np.zeros(shape=(6,1))
    for n in range(6):
        for m in range(6):
            new_img1[n,m]=np.sum(img[n:n+2,m:m+2])/9
        new_img2[n,:]=np.sum(new_img1[n])
return new_img2.reshape((1,6))    #返回1*6行压缩矩阵

#赋值数据集，固定为8*8
digits,targets = load_digits(return_X_y=True)
digits_dict={0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[]}
for item,label in zip(digits,targets):
    img = item.reshape((8,8))
    img3 = feature(img)
digits_dict.get(label).append(img3)

#标签为0的第一个样本的特征向量显示为1*6
print(digits_dict.get(0)[0])

#显示以标签分类的每个样本的特征向量长度
for label,item in digits_dict.items():
    print(label,':',len(item))

#存盘
filename='digits_dict.npy'
np.save(filename,digits_dict)

#读盘
data=np.load(filename,allow_pickle=True)
data = data.item()

#显示读盘后的数据
for label,item in data.items():
print(label,':',len(item))
```

 **实验结果** 

![输入图片说明](https://images.gitee.com/uploads/images/2021/0316/160119_bbbc80dc_8794964.png "屏幕截图.png")

 **分析** 

从给出的程序可以看到，feature函数将输入的8*8矩阵平分为36个3*3小矩阵，计算每个小矩阵的灰度平均值，所得特征向量矩阵应为6*6形式，再经过行变换压缩，矩阵转变为6*1形式，最后返回的是1*6形式的特征压缩矩阵。根据实验结果图可知，第一个样本的特征压缩矩阵的形式符合程序设计要求，共一行6列。输出的其他数据分别为存盘和取盘时0-9标签各占有的样本个数，也是特征矩阵的个数，因此个数之和为1797。
