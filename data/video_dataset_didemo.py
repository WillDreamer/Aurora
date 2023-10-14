from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url
from PIL import Image
import torch
import numpy as np
import random
import decord
from decord import VideoReader
import json
import os
from data.utils import pre_caption
import torchvision.transforms as transforms
from PIL import Image
import io
from torch.utils.data.dataloader import default_collate
import av
import torch
import numpy as np
import lmdb
import random
import ujson as json
from collections import defaultdict
decord.bridge.set_bridge("torch")
from .utils import decode
import math
import numpy as np
import random
import torch



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



def load_jsonl(filename):
    with open(filename, "r") as f:
        return [json.loads(l.strip("\n")) for l in f.readlines()]


def get_video_decoding_kwargs(container, num_frames, target_fps,
                              num_clips=None, clip_idx=None,
                              sampling_strategy="rand",
                              safeguard_duration=False, video_max_pts=None):
    if num_clips is None:
        three_clip_names = ["start", "middle", "end"]  # uniformly 3 clips
        assert sampling_strategy in ["rand", "uniform"] + three_clip_names
        if sampling_strategy == "rand":
            decoder_kwargs = dict(
                container=container,
                sampling_rate=1,
                num_frames=num_frames,
                clip_idx=-1,  # random sampling
                num_clips=None,  # will not be used when clip_idx is `-1`
                target_fps=target_fps
            )
        elif sampling_strategy == "uniform":
            decoder_kwargs = dict(
                container=container,
                sampling_rate=1,  # will not be used when clip_idx is `-2`
                num_frames=num_frames,
                clip_idx=-2,  # uniformly sampling from the whole video
                num_clips=1,  # will not be used when clip_idx is `-2`
                target_fps=target_fps  # will not be used when clip_idx is `-2`
            )
        else:  # in three_clip_names
            decoder_kwargs = dict(
                container=container,
                sampling_rate=1,
                num_frames=num_frames,
                clip_idx=three_clip_names.index(sampling_strategy),
                num_clips=3,
                target_fps=target_fps
            )
    else:  # multi_clip_ensemble, num_clips and clip_idx are only used here
        assert clip_idx is not None
        # sampling_strategy will not be used, as uniform sampling will be used by default.
        # uniformly sample `num_clips` from the video,
        # each clip sample num_frames frames at target_fps.
        decoder_kwargs = dict(
            container=container,
            sampling_rate=1,
            num_frames=num_frames,
            clip_idx=clip_idx,
            num_clips=num_clips,
            target_fps=target_fps,
            safeguard_duration=safeguard_duration,
            video_max_pts=video_max_pts
        )
    return decoder_kwargs

def extract_frames_from_video_binary(
        in_mem_bytes_io, target_fps=3, num_frames=3, num_clips=None, clip_idx=None,
        multi_thread_decode=False, sampling_strategy="rand",
        safeguard_duration=False, video_max_pts=None):
    """
    Args:
        in_mem_bytes_io: binary from read file object
            >>> with open(video_path, "rb") as f:
            >>>     input_bytes = f.read()
            >>> frames = extract_frames_from_video_binary(input_bytes)
            OR from saved binary in lmdb database
            >>> env = lmdb.open("lmdb_dir", readonly=True)
            >>> txn = env.begin()
            >>> stream = io.BytesIO(txn.get(str("key").encode("utf-8")))
            >>> frames = extract_frames_from_video_binary(stream)
            >>> from torchvision.utils import save_image
            >>> save_image(frames[0], "path/to/example.jpg")  # save the extracted frames.
    Returns:
        torch.uint8, (T, C, H, W)
    """
    try:
        video_container = av.open(in_mem_bytes_io, metadata_errors="ignore")
    except Exception as e:
        return None, None

    if multi_thread_decode:
        # Enable multiple threads for decoding.
        video_container.streams.video[0].thread_type = "AUTO"
    # (T, H, W, C), channels are RGB
    # see docs in decoder.decode for usage of these parameters.
    decoder_kwargs = get_video_decoding_kwargs(
        container=video_container, num_frames=num_frames,
        target_fps=target_fps, num_clips=num_clips, clip_idx=clip_idx,
        sampling_strategy=sampling_strategy,
        safeguard_duration=safeguard_duration, video_max_pts=video_max_pts)
    frames, video_max_pts = decode(**decoder_kwargs)
    # (T, H, W, C) -> (T, C, H, W)
    if frames is not None:
        frames = frames.permute(0, 3, 1, 2)
    return frames, video_max_pts

def mk_video_ret_datalist(raw_datalist, data_ratio = 1.0):
    """
    Args:
        raw_datalist: list(dict)
        cfg:
    Returns:
    """
    if data_ratio != 1.0:
        random.shuffle(raw_datalist)
        raw_datalist = raw_datalist[:int(len(raw_datalist) * data_ratio)]

    datalist = []
    qid = 0
    for raw_d in raw_datalist:
        d = dict(
            id=qid,
            txt=raw_d["caption"],
            vid_id=raw_d["clip_name"]
        )
        qid += 1
        datalist.append(d)
    return datalist

def flat_list_of_lists(l):
    """flatten a list of lists [[1,2], [3,4]] to [1,2,3,4]"""
    return [item for sublist in l for item in sublist]

def chunk_list(examples, chunk_size=2, pad_to_divisible=True):
    """
    Args:
        examples: iterable, examples grouped by image/video
        chunk_size: int, number of examples in each chunk.
        pad_to_divisible: bool, pad the examples to be divisible by chunk_size.
    >>> test_examples = [3, 4, 5, 6, 7]
    >>> chunk_list(test_examples, chunk_size=2, pad_to_divisible=True)
    [[3, 4], [5, 6], [7, 7]]  # the lst element has some randomness
    >>> chunk_list(test_examples, chunk_size=2, pad_to_divisible=False)
    [[3, 4], [5, 6], [7]]
    """
    n_examples = len(examples)
    remainder = n_examples % chunk_size
    if pad_to_divisible and remainder > 0:
        n_pad = chunk_size - remainder
        pad = random.choices(examples, k=n_pad)  # with replacement
        examples = examples + pad
        n_examples = len(examples)
        remainder = 0
    chunked_examples = []
    n_chunks = int(n_examples / chunk_size)
    n_chunks = n_chunks + 1 if remainder > 0 else n_chunks
    for i in range(n_chunks):
        chunked_examples.append(examples[i*chunk_size: (i+1)*chunk_size])
    return chunked_examples

def mk_input_group(key_grouped_examples, max_n_example_per_group=2, is_train=True,
                   example_unique_key=None):
    """ Re-organize examples into groups. Each input group will have a single image paired
    with X (X=max_n_example_per_img) examples. Images with total #examples > X will be
    split into multiple groups. In the case a group has < X examples, we will copy
    the examples to make the group has X examples.
    Args:
        key_grouped_examples: dict, each key is image/video id,
            each value is a list(example) associated with this image/video
        max_n_example_per_group: int, pair max #examples with each image/video.
           Note that each image can have multiple groups.
        is_train: bool, if True, copy the examples to make sure each input
            group has max_n_example_per_group examples.
        example_unique_key: str, used to make sure no inputs are discarded by matching
            the input and output ids specified by `example_unique_key`
    """
    input_groups = []  # each element is (id, list(example))
    for k, examples in key_grouped_examples.items():
        chunked_examples = chunk_list(examples,
                                      chunk_size=max_n_example_per_group,
                                      pad_to_divisible=is_train)
        for c in chunked_examples:
            # if len(c) == 0:
            #     continue
            input_groups.append((k, c))

    if example_unique_key is not None:
        print(f"Using example_unique_key {example_unique_key} to check whether input and output ids m")
        # sanity check: make sure we did not discard any input example by accident.
        input_question_ids = flat_list_of_lists(
            [[sub_e[example_unique_key] for sub_e in e] for e in key_grouped_examples.values()])
        output_question_ids = flat_list_of_lists(
            [[sub_e[example_unique_key] for sub_e in e[1]] for e in input_groups])
        assert set(input_question_ids) == set(output_question_ids), "You are missing "
    return input_groups


class DidemoDataset_train(Dataset):
    def __init__(self, video_root, ann_root, num_frm=8, frm_sampling_strategy="rand", max_img_size=224, video_fmt='.mp4'):
        '''
        image_root (string): Root directory of video
        ann_root (string): directory to store the annotation file
        '''        
        # url = 'https://storage.googleapis.com/sfr-vision-language-research/datasets/msrvtt_test.jsonl'
        filename = 'train.jsonl'

        # download_url(url,ann_root)
        self.annotation = mk_video_ret_datalist(load_jsonl(os.path.join(ann_root,filename)))

        self.multi_thread_decode = False

        self.grouped = defaultdict(list)  # examples grouped by image/video id
        for d in self.annotation:
            self.grouped[d["vid_id"]].append(d)

        self.group_datalist = mk_input_group(
            self.grouped,
            max_n_example_per_group=1,  # force 1 in eval
            is_train=True)
        

        self.num_frm = num_frm
        self.frm_sampling_strategy = frm_sampling_strategy
        self.max_img_size = max_img_size
        self.video_root = video_root
        self.video_fmt = video_fmt
        self.train_transformer = transforms.Compose([
            transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor()])


        self.text = [pre_caption(ann[1][0]['txt'],40) for ann in self.group_datalist]
        self.txt2video = [i for i in range(len(self.group_datalist))]
        self.video2txt = self.txt2video           

        self.video_ids = {}  
        n = 0
        for ann in self.group_datalist:
            img_id = ann[0]
            if img_id not in self.video_ids.keys():
                self.video_ids[img_id] = n
                n += 1           
            
            
    def __len__(self):
        return len(self.group_datalist)

    def __getitem__(self, index):
        # some of the videos are missing
        for i in range(3):
            vid_id, examples = self.group_datalist[index]
            env = lmdb.open(self.video_root, readonly=True,create=False,lock=False)
            txn = env.begin(buffers=True)
            io_stream = io.BytesIO(txn.get(str(vid_id).encode("utf-8")))
            frames, video_max_pts = extract_frames_from_video_binary(io_stream, target_fps=3, num_frames=self.num_frm,multi_thread_decode=self.multi_thread_decode,sampling_strategy=self.frm_sampling_strategy,safeguard_duration=False)
            if frames is None:
                index = random.randint(0, len(self.group_datalist) - 1)
                continue
            frames = torch.cat([self.train_transformer(frames[i]).unsqueeze(0) for i in range(frames.shape[0])])
            return frames, self.text[index],self.video_ids[vid_id]
    

   


class DidemoDataset_eval(Dataset):
    def __init__(self, video_root, ann_root, num_frm=8, frm_sampling_strategy="rand", max_img_size=224, video_fmt='.mp4'):
        '''
        image_root (string): Root directory of video
        ann_root (string): directory to store the annotation file
        '''        
        # url = 'https://storage.googleapis.com/sfr-vision-language-research/datasets/msrvtt_test.jsonl'
        filename = 'test.jsonl'

        # download_url(url,ann_root)
        self.annotation = mk_video_ret_datalist(load_jsonl(os.path.join(ann_root,filename)))
        
        self.multi_thread_decode = False

        self.grouped = defaultdict(list)  # examples grouped by image/video id
        for d in self.annotation:
            self.grouped[d["vid_id"]].append(d)

        self.group_datalist = mk_input_group(
            self.grouped,
            max_n_example_per_group=1,  # force 1 in eval
            is_train=True)
        print('video num:', len(self.group_datalist))

        self.num_frm = num_frm
        self.frm_sampling_strategy = frm_sampling_strategy
        self.max_img_size = max_img_size
        self.video_root = video_root
        self.video_fmt = video_fmt
        self.train_transformer = transforms.Compose([
            transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor()])


        self.text = [pre_caption(ann[1][0]['txt'],40) for ann in self.group_datalist]
        self.txt2video = [i for i in range(len(self.group_datalist))]

        self.video2txt = self.txt2video               
            
            
    def __len__(self):
        return len(self.group_datalist)

    def __getitem__(self, index):
        
        vid_id, examples = self.group_datalist[index]
        env = lmdb.open(self.video_root, readonly=True,create=False)
        txn = env.begin(buffers=True)
        io_stream = io.BytesIO(txn.get(str(vid_id).encode("utf-8")))
        frames, video_max_pts = extract_frames_from_video_binary(io_stream, target_fps=3, num_frames=self.num_frm,multi_thread_decode=self.multi_thread_decode,sampling_strategy=self.frm_sampling_strategy,safeguard_duration=False)
        frames = torch.cat([self.train_transformer(frames[i]).unsqueeze(0) for i in range(frames.shape[0])])
        
        return frames, vid_id
