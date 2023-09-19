# flake8: noqa
import os
import lmdb
import numpy as np
import torch
import torch.utils.data as data
import json
import pickle
import cv2

class Data_Video:
    def __init__(self, video):
        self.frames = video.shape[0]
        self.size = video.shape[1:3]
        self.channels = video.shape[3]
        self.video = video.tobytes()

    def get_video(self):
        video = np.frombuffer(self.video, dtype = np.uint8)
        return video.reshape(self.frames, *self.size, self.channels)


class Video_Dataset(data.Dataset):
    def __init__(self, lmdb_video_path, video_keys_list_path, json_path, clip_sec, max_clip_num):
        """
        :param lmdb_video_path: the path to read the video lmdb(directory)
        :param video_keys_list_path: the path to read the txt file containing a list of video ids
        :param json_path: the path to read the file of pair dict, the format is {vid:[mid,tag]}
        """

        self.lmdb_video_path = lmdb_video_path
        self.env_video = lmdb.open(self.lmdb_video_path, subdir=True, readonly=True, lock=False, meminit=False)
        self.clip_sec = clip_sec
        self.max_clip_num = max_clip_num

        # get json content
        f = open(json_path, 'r')
        self.v_pre_pairs = json.load(f)
        self.v_pre_keys = []
        with open(video_keys_list_path, 'r') as f:
            for key in f:
                self.v_pre_keys.append(key.strip('\n'))

        self.v_pre_length = len(self.v_pre_keys)
        self.v_pre_targets = [self.v_pre_pairs[vid][1] for vid in self.v_pre_keys]
        self.targets_set = set(self.v_pre_targets)  # remove the extra target
        self.target_to_indices = {target: np.where(np.array(self.v_pre_targets) == target)[0] \
                                  for target in self.targets_set}


    def __getitem__(self, index):
        def normalize_video_longer_than_four_sec(video, video_sec, clip_sec):
            remainder = video_sec - video_sec // clip_sec * clip_sec  # check the number of frames which could not be put into a complete clip
            if remainder <= clip_sec / 2:  # remainder duration less than clip_sec/2 seconds, abandon remainder
                video_complete = video[: int(video_sec // clip_sec * clip_sec)]
            else:
                video_add_length = (video_sec // clip_sec +1) * clip_sec - video_sec
                video_complete = np.vstack((video, np.repeat(video[-1:], video_add_length, 0)))
            return video_complete
        vid = self.v_pre_keys[index]
        with self.env_video.begin(write=False) as txn_video:
            video_data_bytes = txn_video.get(vid.encode())
        video = pickle.loads(video_data_bytes)
        video_data = video.get_video()
        video_gray = []
        for i in range(video_data.shape[0]):
            video_gray.append(cv2.cvtColor(video_data[i], cv2.COLOR_BGR2GRAY))
        # video_data = np.transpose(video_data, [0, 3, 1, 2])
        video_gray = np.stack(video_gray, axis=0)  # [frame, height, width]
        frame_num = video_gray.shape[0]
        # process the video_data
        video_complete = video_gray
        assert vid in self.v_pre_pairs, "Invalid video id! Not in the video-music pair dict!"
        _, target, label = self.v_pre_pairs[vid]  # Use video id to search music id in video-music pair dict as well as the target
        if frame_num <= self.clip_sec / 2:   # audio duration less than clip_sec/2 seconds, abandon audio
            return np.array([]), np.array([])
        elif self.clip_sec / 2 < frame_num <= self.clip_sec:
            frame_add_length = int(self.clip_sec - frame_num)
            video_complete = np.vstack((video_gray, np.repeat(video_gray[-1:], frame_add_length, 0)))
        else:
            if frame_num >= self.max_clip_num:
                video_complete = video_gray[ : self.max_clip_num]
            video_complete = normalize_video_longer_than_four_sec(video_complete, video_complete.shape[0], self.clip_sec) # 4 dimensions darray

        # devide into different clips
        video_frame = video_complete.shape[0]
        video_final = []
        i = 0
        tmp_video = []
        while ((video_frame-i) // self.clip_sec) != 0:
            for j in range(i, i + 4):
                tmp_video.append(video_complete[j])
            tmp_video = np.concatenate(tmp_video, axis= 1)
            video_final.append(tmp_video)
            tmp_video = []
            i += 4
        video_final = np.stack(video_final, axis=0)

        # tile the p_target
        label = np.repeat(int(label), video_complete.shape[0]//self.clip_sec).reshape(-1, 1)
        return video_final, label

    def __len__(self):
        return self.v_pre_length


def thread_loaddata(video_feature_database_path, video_index_file_path, json_path, clip_sec = 4, file_num = 512, max_clip_num = 30):
    video_feature_database_path = os.path.abspath(video_feature_database_path)
    dataset = Video_Dataset(video_feature_database_path, video_index_file_path, json_path, clip_sec, max_clip_num)
    loader = data.DataLoader(
        dataset=dataset,
        batch_size=file_num,
        shuffle=True,
        num_workers=8,
        collate_fn=collate_data,
    )

    return loader


def collate_data(batch):
    video_feature, label = zip(*batch)
    video_list = []
    label_list = []
    for i in range(len(video_feature)):
        if np.array_equal(video_feature[i], np.array([])):
            continue
        video_list.append(video_feature[i])
        label_list.append(label[i])
    video_numpy = np.concatenate(video_list, axis=0)
    label_numpy = np.concatenate(label_list, axis=0)
    video_tensor = torch.FloatTensor(video_numpy)
    label_tensor = torch.LongTensor(label_numpy)
    return video_tensor, label_tensor
