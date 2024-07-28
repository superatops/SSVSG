## Paper "Sentinel mechanism for Visual Semantic Graph-based image captioning"



### Requirements

python 2.7.15

torch 1.0.1

Specific conda env is shown in ezs.yml

BTW, you need to download [coco-captions](https://github.com/tylin/coco-caption) and [cider](https://github.com/vrama91/cider) folder in this directory for evaluation.

### Data Files and Models

1. Download and add files in data directory in [google drive](https://drive.google.com/drive/folders/1VYeFocLMz2msICHu8DFRWBwTKcok7VAe?usp=sharing) or [baidu netdisk](链接：https://pan.baidu.com/s/1ddtfdlwD65cm4JmVu6GF3w 
提取码：39pa) to data directory here. 

2.Download preprocessed coco captions from link from Karpathy's homepage. The do:
```
python scripts/prepro_labels.py --input_json data/dataset_coco.json --output_json data/cocotalk.json --output_h5 data/cocotalk
```

Download the file 'cocobu2.json' and 'cocobu2_label.h5' from https://drive.google.com/drive/folders/1GvwpchUnfqUjvlpWTYbmEvhvkJTIWWRb?usp=sharing and put them into the folder 'data' (if you do not have this folder, just create one), which are processed by myself for facilitating the usage of this code. I also release two well-trained models based on these two files which are modelid740072 and modelid640075.

3.Download pre-extracted feature from https://github.com/peteanderson80/bottom-up-attention. 

```
mkdir data/bu_data; cd data/bu_data
wget https://storage.googleapis.com/bottom-up-attention/trainval.zip
unzip trainval.zip
```
Then :
```
python script/make_bu_data.py --output_dir data/cocobu
```

4.Download the files 'coco_pred_sg.zip' and 'coco_spice_sg2.zip' from https://drive.google.com/drive/folders/1GvwpchUnfqUjvlpWTYbmEvhvkJTIWWRb?usp=sharing and add them into the folder 'data' and then unzip them. 


### Scripts

MLE training:

`python train.py --gpus 0 --id train`

RL training

`python train.py --gpus 0 --id train-rl --learning_rate 2e-5 --resume_from train --resume_from_best True --self_critical_after 0 --max_epochs 60 --learning_rate_decay_start -1 --scheduled_sampling_start -1 --reduce_on_plateau`

Evaluate your own model or Load trained model:

`python eval.py --gpus 0 --resume_from train`

and

`python eval.py --gpus 0 --resume_from train-rl`










