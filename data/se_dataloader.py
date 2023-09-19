# flake8: noqa
import os
import lmdb
import numpy as np
import torch
import torch.utils.data as data
import json
import pickle, copy
from .samplerfactory import SamplerFactory

class Audio_feature:
    def __init__(self, audio):
        self.clip = audio.shape[0]
        self.dim = audio.shape[1]
        self.audio = audio.tobytes()

    def get_audio(self):
        audio = np.frombuffer(self.audio, dtype = np.float32)
        return audio.reshape(self.clip, self.dim)

class Video_feature:
    def __init__(self, video):
        self.clip = video.shape[0]
        self.dim = video.shape[1]
        self.video = video.tobytes()

    def get_video(self):
        video = np.frombuffer(self.video, dtype = np.float32)
        return video.reshape(self.clip, self.dim)

class PretrainedDataset(data.Dataset):
    def __init__(self, lmdb_audio_path, lmdb_video_path, lmdb_text_path, video_keys_list_path, json_path, data_split, audio_rhythm_pkl, video_optical_json, max_clip=7):
        """
        :param lmdb_audio_path: the path to read the audio lmdb(directory)
        :param lmdb_video_path: the path to read the video lmdb(directory)
        :param lmdb_text_path: the path to read the text lmdb(directory)
        :param video_keys_list_path: the path to read the txt file containing a list of video ids
        :param json_path: the path to read the file of pair dict, the format is {vid:[mid,tag]}
        :param data_split: denote the data split, train/train_val/val/test
        """

        self.lmdb_audio_path = lmdb_audio_path
        self.lmdb_video_path = lmdb_video_path
        self.lmdb_text_path = lmdb_text_path
        self.env_audio = lmdb.open(self.lmdb_audio_path, subdir=True, readonly=True, lock=False, meminit=False)
        self.env_video = lmdb.open(self.lmdb_video_path, subdir=True, readonly=True, lock=False, meminit=False)
        self.env_text = lmdb.open(self.lmdb_text_path, subdir=True, readonly=True, lock=False, meminit=False)
        self.max_clip = max_clip

        # get rhythm for audio
        with open(audio_rhythm_pkl, 'rb') as f_r_audio:
            self.audio_rhythm_dict = pickle.load(f_r_audio)
        # get optical flow for video
        with open(video_optical_json, 'rb') as f_r_video:
            self.video_optical_dict = pickle.load(f_r_video)

        # get json content
        with open(json_path, 'r') as f:
            self.pairs = json.load(f)
        self.keys = []
        with open(video_keys_list_path, 'r') as f:
            for key in f:
                self.keys.append(key.strip('\n'))

        self.length = len(self.keys)
        self.targets = [self.pairs[vid][1] for vid in self.keys]
        self.targets_set = set(self.targets)  # remove the extra target
        self.target_to_indices = {target: np.where(np.array(self.targets) == target)[0] \
                                  for target in self.targets_set}
        self.data_split = data_split

        if self.data_split == "val":
            random_state = np.random.RandomState(29)
            pos_neg_pairs = [
                [
                    i, random_state.choice(self.target_to_indices[
                                               np.random.choice(
                                                   list(self.targets_set - set([self.targets[i]]))
                                               )
                                           ])
                ]
                for i in range(len(self.targets))
            ]
            self.val_pairs = pos_neg_pairs

    def __getitem__(self, index):
        def do_max_clip(feature, rhythm, type):
            feature_complete = copy.deepcopy(feature)
            feature_rhythm = copy.deepcopy(rhythm)
            if feature.shape[0] < self.max_clip:
                remainder = self.max_clip - feature.shape[0]
                former_clip_num = feature.shape[0]
                if type == "audio":
                    if remainder >= former_clip_num:
                        feature_complete = np.vstack((feature_complete, np.repeat(feature, remainder//former_clip_num, 0)))
                        feature_rhythm += rhythm * (remainder // former_clip_num)
                    i = 0
                    remainder = self.max_clip - feature_complete.shape[0]
                    while remainder > 0:
                        feature_complete = np.vstack((feature_complete, feature[i:i+1]))
                        feature_rhythm += rhythm[i:i + 1]
                        remainder -= 1
                        i += 1
                elif type == "video":
                    feature_complete = np.vstack((feature_complete, np.repeat(feature[-1:], remainder, 0)))
                    feature_rhythm += [rhythm[-1]] * remainder
            elif feature.shape[0] > self.max_clip:
                feature_complete = feature[ : self.max_clip]
                feature_rhythm = rhythm[ :self.max_clip]

            return feature_complete, feature_rhythm

        p_vid = self.keys[index]
        with self.env_video.begin(write=False) as txn_video:
            p_video_feature_bytes = txn_video.get(p_vid.encode())
            p_video_feature_bytes = pickle.loads(p_video_feature_bytes)
            p_video_feature = p_video_feature_bytes.get_video()
        p_video_optical = self.video_optical_dict[p_vid]

        assert p_vid in self.pairs, "Invalid video id! Not in the video-music pair dict!"

        # get positive sample
        p_aid, p_target, p_label = self.pairs[p_vid]  # Use video id to search music id in video-music pair dict as well as the target
        with self.env_audio.begin(write=False) as txn_audio:
            p_audio_feature_bytes = txn_audio.get(p_aid.encode())
            p_audio_feature_bytes = pickle.loads(p_audio_feature_bytes)
            p_audio_feature = p_audio_feature_bytes.get_audio()
        p_audio_rhythm = self.audio_rhythm_dict[p_aid]
        with self.env_text.begin(write=False) as txn_text:
            p_text_feature_bytes = txn_text.get(p_target.encode())
        p_text_feature = np.frombuffer(p_text_feature_bytes, np.float32)

        # get negative sample
        if self.data_split == "train" or self.data_split == "train_val":
            n_target = np.random.choice(list(self.targets_set - set([p_target])))
            n_index = np.random.choice(self.target_to_indices[n_target])
            n_vid = self.keys[n_index]
            with self.env_video.begin(write=False) as txn_video:
                n_video_feature_bytes = txn_video.get(n_vid.encode())
                n_video_feature_bytes = pickle.loads(n_video_feature_bytes)
                n_video_feature = n_video_feature_bytes.get_video()
            n_video_optical = self.video_optical_dict[n_vid]
            assert n_vid in self.pairs, "Invalid video id! Not in the video-music pair dict!"
            n_aid, n_target, n_label = self.pairs[n_vid]
            with self.env_audio.begin(write=False) as txn_audio:
                n_audio_feature_bytes = txn_audio.get(n_aid.encode())
                n_audio_feature_bytes = pickle.loads(n_audio_feature_bytes)
                n_audio_feature = n_audio_feature_bytes.get_audio()
            n_audio_rhythm = self.audio_rhythm_dict[n_aid]
            with self.env_text.begin(write=False) as txn_text:
                n_text_feature_bytes = txn_text.get(n_target.encode())
            n_text_feature = np.frombuffer(n_text_feature_bytes, np.float32)
        elif self.data_split == "val":
            n_index = self.val_pairs[index][1]
            n_vid = self.keys[n_index]
            with self.env_video.begin(write=False) as txn_video:
                n_video_feature_bytes = txn_video.get(n_vid.encode())
                n_video_feature_bytes = pickle.loads(n_video_feature_bytes)
                n_video_feature = n_video_feature_bytes.get_video()
            n_video_optical = self.video_optical_dict[n_vid]
            assert n_vid in self.pairs, "Invalid video id! Not in the video-music pair dict!"
            n_aid, n_target, n_label = self.pairs[n_vid]
            with self.env_audio.begin(write=False) as txn_audio:
                n_audio_feature_bytes = txn_audio.get(n_aid.encode())
                n_audio_feature_bytes = pickle.loads(n_audio_feature_bytes)
                n_audio_feature = n_audio_feature_bytes.get_audio()
            n_audio_rhythm = self.audio_rhythm_dict[n_aid]
            with self.env_text.begin(write=False) as txn_text:
                n_text_feature_bytes = txn_text.get(n_target.encode())
            n_text_feature = np.frombuffer(n_text_feature_bytes, np.float32)

        # preprocess audio and video feature, copy till equals to max_cip

        p_video_feature, p_video_optical = do_max_clip(p_video_feature, p_video_optical, "video")

        p_audio_feature, p_audio_rhythm = do_max_clip(p_audio_feature, p_audio_rhythm, "audio")

        n_video_feature, n_video_optical = do_max_clip(n_video_feature, n_video_optical, "video")

        n_audio_feature, n_audio_rhythm = do_max_clip(n_audio_feature, n_audio_rhythm, "audio")

        return p_audio_feature, p_video_feature, n_audio_feature, n_video_feature, p_aid, p_vid, n_aid, n_vid, p_target, n_target, p_text_feature, n_text_feature, int(
            p_label), int(n_label), np.array(p_video_optical), np.array(p_audio_rhythm), np.array(n_video_optical), np.array(n_audio_rhythm)

    def __len__(self):
        return self.length


def thread_loaddata(audio_feature_database_path, video_feature_database_path, text_feature_database_path, file_num,
                    video_index_file_path, json_path, logger, data_split, audio_rhythm_pkl, video_optical_json, max_clip):
    audio_feature_database_path = os.path.abspath(audio_feature_database_path)
    video_feature_database_path = os.path.abspath(video_feature_database_path)
    text_feature_database_path = os.path.abspath(text_feature_database_path)
    dataset = PretrainedDataset(audio_feature_database_path, video_feature_database_path, text_feature_database_path,
                          video_index_file_path, json_path, data_split, audio_rhythm_pkl, video_optical_json, max_clip)
    if data_split == "train" or data_split == "train_val":
        n_batches = 550
    else:
        n_batches = 500
    sampler = SamplerFactory(logger).get(
        class_idxs=list(dataset.target_to_indices.values()),
        batch_size=file_num,
        n_batches=n_batches,
        alpha=1,
        kind='fixed'
    )
    loader = data.DataLoader(
        dataset=dataset,
        batch_sampler=sampler,
        num_workers=20,
        prefetch_factor=2,
        pin_memory=True,
        collate_fn=collate_data,
    )
    return loader


def collate_data(batch):
    p_audio_feature, p_video_feature, n_audio_feature, n_video_feature, p_aid, p_vid, n_aid, n_vid, p_target, n_target, p_text_feature, n_text_feature, p_label, n_label, p_video_optical, p_audio_rhythm, n_video_optical, n_audio_rhythm = zip(*batch)
    p_audio_list = []
    p_video_list = []
    n_audio_list = []
    n_video_list = []
    p_aid_list = []
    p_vid_list = []
    n_aid_list = []
    n_vid_list = []
    p_target_list = []
    n_target_list = []
    p_text_list = []
    n_text_list = []
    p_label_list = []
    n_label_list = []
    p_optical_list = []
    p_rhythm_list = []
    n_optical_list = []
    n_rhythm_list = []
    for i in range(len(p_audio_feature)):
        p_audio_list.append(p_audio_feature[i])
        p_video_list.append(p_video_feature[i])
        n_audio_list.append(p_audio_feature[i])
        n_video_list.append(p_video_feature[i])
        p_aid_list.append(p_aid[i])
        p_vid_list.append(p_vid[i])
        n_aid_list.append(n_aid[i])
        n_vid_list.append(n_vid[i])
        p_target_list.append(p_target[i])
        n_target_list.append(n_target[i])
        p_text_list.append(p_text_feature[i])
        n_text_list.append(n_text_feature[i])
        p_label_list.append(p_label[i])
        n_label_list.append(n_label[i])
        p_optical_list.append(p_video_optical[i])
        p_rhythm_list.append(p_audio_rhythm[i])
        n_optical_list.append(n_video_optical[i])
        n_rhythm_list.append(n_audio_rhythm[i])
    p_audio_numpy = np.stack(p_audio_list, axis=0)
    p_video_numpy = np.stack(p_video_list, axis=0)
    p_aid_numpy = np.vstack(p_aid_list)
    p_vid_numpy = np.vstack(p_vid_list)
    n_audio_numpy = np.stack(n_audio_list, axis=0)
    n_video_numpy = np.stack(n_video_list, axis=0)
    n_aid_numpy = np.vstack(n_aid_list)
    n_vid_numpy = np.vstack(n_vid_list)
    p_target_numpy = np.vstack(p_target_list)
    n_target_numpy = np.vstack(n_target_list)
    p_optical_numpy = np.stack(p_optical_list, axis=0)
    p_rhythm_numpy = np.stack(p_rhythm_list, axis=0)
    n_optical_numpy = np.stack(n_optical_list, axis=0)
    n_rhythm_numpy = np.stack(n_rhythm_list, axis=0)
    p_audio_tensor = torch.FloatTensor(p_audio_numpy)
    p_video_tensor = torch.FloatTensor(p_video_numpy)
    n_audio_tensor = torch.FloatTensor(n_audio_numpy)
    n_video_tensor = torch.FloatTensor(n_video_numpy)
    p_text_tensor = torch.FloatTensor(p_text_list)
    n_text_tensor = torch.FloatTensor(n_text_list)
    p_label = torch.FloatTensor(p_label_list)
    n_label = torch.FloatTensor(n_label_list)
    p_optical_tensor = torch.IntTensor(p_optical_numpy)
    p_rhythm_tensor = torch.IntTensor(p_rhythm_numpy)
    n_optical_tensor = torch.IntTensor(n_optical_numpy)
    n_rhythm_tensor = torch.IntTensor(n_rhythm_numpy)
    return p_audio_tensor, p_video_tensor, n_audio_tensor, n_video_tensor, p_aid_numpy, p_vid_numpy, n_aid_numpy, n_vid_numpy, p_target_numpy, n_target_numpy, p_text_tensor, n_text_tensor, p_label, n_label, p_rhythm_tensor, p_optical_tensor, n_rhythm_tensor, n_optical_tensor

