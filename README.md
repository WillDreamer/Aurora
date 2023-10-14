# [NeurIPS2023] AURORA: Multimodal Parameter-efficient Transfer Learning Based on Mode Approximation

## Introduction
Aurora is an efficient PETL method used in multimodal large model fields. It uses mode approximation to further reduce the trainable parameters and promote the fusion of different modalities.

**1. Comparison with other PETL methods**
![image](https://github.com/xinlong-yang/Aurora/assets/73691354/33bbadb8-cdcc-4105-94fb-ee4fb6b77d00)

**2. Overall architecture**
![image](https://github.com/xinlong-yang/Aurora/assets/73691354/16ae4930-c44d-45c8-95e0-766bc60bb290)

## Getting Started

- Python3, PyTorch>=1.8.0, torchvision>=0.7.0 are required for the current codebase.

#### Image-text Retrieval
- Download COCO and Flickr30k datasets from the original websites, and set 'image_root' in configs/retrieval_{dataset}.yaml accordingly.

- To parameter-efficient finetune on MSCOCO/Flickr:
<pre>python -m torch.distributed.run --nproc_per_node=8 train_retrieval.py --config ./configs/retrieval_{coco, flickr}.yaml --output_dir output/{coco, flickr} </pre> 
- To evaluate on MSCOCO/Flickr:
<pre>python -m torch.distributed.run --nproc_per_node=8 train_retrieval.py --config ./configs/retrieval_{coco, flickr}.yaml --output_dir output/{coco, flickr} --evaluate </pre> 

#### Visual Question Answerring
- Download VQA v2 dataset and Visual Genome dataset from the original websites, and set 'vqa_root' and 'vg_root' in configs/vqa.yaml.

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
