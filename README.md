# RGB-T Multi-Modal Crowd Counting Based on Transformer

The paper has been accepted by BMVC 2022.

The details are in https://bmvc2022.mpi-inf.mpg.de/0427.pdf

## Code
链接：https://pan.baidu.com/s/1sYlFQXqGiY8ykOpehH_kkQ 
提取码：jrux 


## Pretraining Parameter



If you have any question, please email  liuzywen@ahu.edu.cn


## Install dependencies
torch >= 1.0 torchvision opencv numpy scipy, all the dependencies can be easily installed by pip or conda

This code was tested with python 3.8.

install MultiScaleDeformableAttention:
```
cd ./models/ops     
sh ./make.sh
```

## Preprocessing
File predataset_RGBT_CC.py is used to process RGBT-CC datasets to obtain pictures of different sizes.
```
python predataset_RGBT_CC.py
```

## Training
Edit this file for training BL-based IADM model.

training hyperparameters as:
```
--data-dir = ""  # train datasets path.
--save-dir = ""  # save model path.
--pretrained_model = ""  # pre-trained parameters
--max-epoch = 500
--val-epoch = 1
--val_start = 30
--batch-size = 16

```

```
bash train.sh
```

## Testing
Edit this file for testing models.
testing hyperparameters as :
```
--data-dir = ""  # test datasets path
--save-dir = ""  # saved model path
--model = ""  # best_model_XXXXX.pth
```

```
bash test.sh
```


