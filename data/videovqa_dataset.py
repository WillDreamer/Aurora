import os
import json
import random
import torch
import numpy as np
from decord import VideoReader
from torch.utils.data import Dataset
from data.utils import pre_question
from torchvision.datasets.utils import download_url
import random
import decord
from decord import VideoReader


class ImageNorm(object):
    """Apply Normalization to Image Pixels on GPU
    """
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)
        
    def __call__(self, img):

        if torch.max(img) > 1 and self.mean.max() <= 1:
            img.div_(255.)
        return img.sub_(self.mean).div_(self.std)


class videovqa_dataset(Dataset):
    def __init__(self, ann_root='MSRVTT-QA/', video_root = 'MSRVTT/videos/', train_files=[], split="train"):
        self.split = split
        self.max_img_size=224
        self.img_norm = ImageNorm(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        self.video_root = video_root
        self.num_frm = 8
        self.frm_sampling_strategy = 'rand'

        if split=='train':
            with open(f"{ann_root}train_qa.json") as f:
                self.annotation = json.load(f)

        else:
            with open(f"{ann_root}test_qa.json") as f:
                self.annotation = json.load(f)
                
        
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    
        
        video_path = self.video_root + 'video' + str(self.annotation[index]['video_id']) + '.mp4' 
        vid_frm_array = self._load_video_from_path_decord(video_path, height=self.max_img_size, width=self.max_img_size)
        video = self.img_norm(vid_frm_array.float())
        
        question = self.annotation[index]['question']                    
        
        answer = self.annotation[index]['answer']         
        
        if self.split == 'test':
            return video, question, self.annotation[index]['id']

        elif self.split=='train':                       
            answers = [answer]
            weights = [0.05]  

            return video, question, answers, weights

    def _load_video_from_path_decord(self, video_path, height=None, width=None, start_time=None, end_time=None, fps=-1):
        try:
            if not height or not width:
                vr = VideoReader(video_path)
            else:
                vr = VideoReader(video_path, width=width, height=height)

            vlen = len(vr)

            if start_time or end_time:
                assert fps > 0, 'must provide video fps if specifying start and end time.'

                start_idx = min(int(start_time * fps), vlen)
                end_idx = min(int(end_time * fps), vlen)
            else:
                start_idx, end_idx = 0, vlen

            if self.frm_sampling_strategy == 'uniform':
                frame_indices = np.arange(start_idx, end_idx, vlen / self.num_frm, dtype=int)
            elif self.frm_sampling_strategy == 'rand':
                frame_indices = sorted(random.sample(range(vlen), self.num_frm))
            elif self.frm_sampling_strategy == 'headtail':
                frame_indices_head = sorted(random.sample(range(vlen // 2), self.num_frm // 2))
                frame_indices_tail = sorted(random.sample(range(vlen // 2, vlen), self.num_frm // 2))
                frame_indices = frame_indices_head + frame_indices_tail
            else:
                raise NotImplementedError('Invalid sampling strategy {} '.format(self.frm_sampling_strategy))

            raw_sample_frms = vr.get_batch(frame_indices)
        except Exception as e:
            return None

        raw_sample_frms = raw_sample_frms.permute(0, 3, 1, 2) # T , 3, 224, 224 

        return raw_sample_frms
        
        
def vqa_collate_fn(batch):
    video_list, question_list, answer_list, weight_list, n = [], [], [], [], []
    for video, question, answer, weights in batch:
        video_list.append(video)
        question_list.append(question)
        weight_list += weights       
        answer_list += answer
        n.append(len(answer))
    return torch.stack(video_list,dim=0), question_list, answer_list, torch.Tensor(weight_list), n        