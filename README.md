# [NeurIPS 2023] AURORA: Multimodal Parameter-efficient Transfer Learning Based on Mode Approximation

## Introduction
Aurora is an efficient PETL method used in multimodal large model fields. It uses mode approximation to further reduce the trainable parameters and promote the fusion of different modalities.

**1. Comparison with other PETL methods**
![image](https://github.com/xinlong-yang/Aurora/assets/73691354/33bbadb8-cdcc-4105-94fb-ee4fb6b77d00)

**2. Overall architecture**
![image](https://github.com/xinlong-yang/Aurora/assets/73691354/16ae4930-c44d-45c8-95e0-766bc60bb290)

## Getting Started
#### Requirements
- Python 3.8, PyTorch>=1.8.0, torchvision>=0.7.0, timm>=0.6.13, numpy>=1.21.0, transformers>=4.27.4 are required for the current codebase.

#### Datasets
##### Image-text Retrieval Task
**COCO2014:** download dataset through https://cocodataset.org/#download, you can use such Linux command [wget -c http://images.cocodataset.org/annotations/annotations_trainval2014.zip] to help you download directly.

**Flickr30k:** download dataset through https://shannon.cs.illinois.edu/DenotationGraph/data/index.html; or you can download through this link: https://pan.baidu.com/s/1r0RVUwctJsI0iNuVXHQ6kA, the password is hrf3.

##### Video-text Retrieval
**MSRVTT:** download the video dataset in https://www.mediafire.com/folder/h14iarbs62e7p/shared, and the corresponding annotation file in https://mega.nz/file/UnRnyb7A#es4XmqsLxl-B7MP0KAat9VibkH7J_qpKj9NcxLh8aHg. 

**DiDemo:** download the dataset through this Github project https://github.com/jpthu17/EMCL.

##### Visual Question Answering Task
**VQAv2:** 

**VideoQA:**
#### Image-text Retrieval
- Download COCO and Flickr30k datasets, and set 'image_root' in configs/retrieval_{dataset}.yaml accordingly.

- To parameter-efficient finetune on MSCOCO/Flickr:
<pre>python -m torch.distributed.run --nproc_per_node=8 train_retrieval.py --config ./configs/retrieval_{coco, flickr}.yaml --output_dir output/{coco, flickr} </pre> 
- To evaluate on MSCOCO/Flickr:
<pre>python -m torch.distributed.run --nproc_per_node=8 train_retrieval.py --config ./configs/retrieval_{coco, flickr}.yaml --output_dir output/{coco, flickr} --evaluate </pre> 

#### Visual Question Answering
- Download VQAv2 dataset and Visual Genome dataset from the original websites, and set 'vqa_root' and 'vg_root' in configs/vqa.yaml.

- To parameter-efficient finetune on VQAv2:
<pre>python -m torch.distributed.run --nproc_per_node=8 train_vqa.py --config ./configs/vqa.yaml --output_dir $static_dir</pre> 
- To evaluate on VQAv2 (need to update the result file to the official server):
<pre>python -m torch.distributed.run --nproc_per_node=8 train_vqa.py --config ./configs/vqa.yaml --output_dir $static_dir --evaluate </pre> 

#### Video-text Retrieval and VideoQA
- To parameter-efficient finetune on MSRVTT:
<pre>python -m torch.distributed.run --nproc_per_node=8 train_video_retrieval.py --config ./configs/retrieval_msrvtt.yaml --output_dir $static_dir</pre> 
- To parameter-efficient finetune on DiDemo:
<pre>python -m torch.distributed.run --nproc_per_node=8 train_video_retrieval.py --config ./configs/retrieval_didemo.yaml --output_dir $static_dir</pre> 
- To parameter-efficient finetune on VideoQA:
<pre>python -m torch.distributed.run --nproc_per_node=8 train_vqa.py --config ./configs/videoqa.yaml --output_dir $static_dir</pre> 

## Acknowledgement
Our codebase is built based on BLIP, timm and transformers. We thank the authors for the nicely organized code!

## How To Cite Aurora
If you use this code in your research, please kindly cite the following paper:
```
@article{wang2023mode,
  title={Mode Approximation Makes Good Vision-Language Prompts},
  author={Wang, Haixin and Yang, Xinlong and Chang, Jianlong and Jin, Dian and Sun, Jinan and Zhang, Shikun and Luo, Xiao and Tian, Qi},
  journal={arXiv preprint arXiv:2305.08381},
  year={2023}
}
```
