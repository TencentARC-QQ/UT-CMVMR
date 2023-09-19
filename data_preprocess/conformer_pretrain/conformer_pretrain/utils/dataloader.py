# flake8: noqa
import os
import copy
import lmdb
import numpy as np
import torch
import torchaudio
import torch.utils.data as data
import json
import pickle

class Data_Audio:
    def __init__(self, audio, duration, sr):
        self.sr = sr
        self.duration = duration
        self.audio = audio.tobytes()

    def get_audio(self):
        audio = np.frombuffer(self.audio, dtype = np.float32)
        return audio

class NormDataset(data.Dataset):
    def __init__(self, lmdb_audio_path, json_path, audio_keys_list_path, clip_sec = 4):
        self.lmdb_audio_path = lmdb_audio_path
        self.env_audio = lmdb.open(self.lmdb_audio_path, subdir = True, readonly = True, lock = False, meminit = False)

        # Train/test aid list
        self.aids = []
        with open(audio_keys_list_path) as f_r:
            for aid in f_r:
                self.aids.append(aid.strip())

        # Prepare for the music id to tag id/music duration dict
        aid2tag_dur = {}
        with open(json_path) as f_r:
            aid2tag_dur = json.load(f_r)
        self.tag_set = set()
        for aid in aid2tag_dur:
            self.tag_set.add(aid2tag_dur[aid][0])

        tag2idx = {}
        idx_count = 0
        for tag in sorted(self.tag_set):
            tag2idx[tag] = idx_count
            idx_count += 1

        self.aid2tagIdx_dur = {}
        for aid in aid2tag_dur:
            self.aid2tagIdx_dur[aid] = [tag2idx[aid2tag_dur[aid][0]], aid2tag_dur[aid][1]]

        self.clip_sec = clip_sec

    def __getitem__(self, index):
        aid = self.aids[index]
        with self.env_audio.begin(write = False) as txn_audio:
            audio_data_bytes = txn_audio.get(aid.encode())
        audio = pickle.loads(audio_data_bytes)
        audio_data = audio.get_audio()
        sr = int(audio.sr)
        duration = int(self.aid2tagIdx_dur[aid][1])
        tagIdx = self.aid2tagIdx_dur[aid][0]
        clip_num = duration // self.clip_sec

        # Compute Filter-bank features, channels set as 80, compatible with the conformer default setting
        audio_data = copy.deepcopy(audio_data)
        audio_data_torch = torch.from_numpy(audio_data).reshape(1, -1)
        fbank_feats = torch.FloatTensor([])
        for i in range(clip_num):
            feat = torchaudio.compliance.kaldi.fbank(
                waveform = audio_data_torch[ :, i * sr * self.clip_sec : (i + 1) * sr * self.clip_sec],
                sample_frequency = sr,
                num_mel_bins = 80,
                htk_compat = True
            )
            feat = feat.unsqueeze(0)
            fbank_feats = torch.cat((fbank_feats, feat), axis = 0)
        fbank_feats = fbank_feats.numpy()

        label = np.repeat(tagIdx, clip_num).reshape(-1, 1)

        return fbank_feats, label

    def __len__(self):
        return len(self.aids)

def thread_loaddata(audio_database_path, json_path, audio_keys_list_path, clip_sec = 4, file_num = 512):
    audio_database_path = os.path.abspath(audio_database_path)
    dataset = NormDataset(audio_database_path, json_path, audio_keys_list_path, clip_sec)
    loader = data.DataLoader(
        dataset = dataset,
        batch_size = file_num,
        shuffle = True,
        num_workers = 8,
        collate_fn = collate_data,
    )
    return loader

def collate_data(batch):
    audio_feature, label = zip(*batch)
    audio_list = []
    label_list = []
    for i in range(len(audio_feature)):
        audio_list.append(audio_feature[i])
        label_list.append(label[i])
    audio_numpy = np.vstack(audio_list)
    label_numpy = np.vstack(label_list)
    permute_idx = np.random.permutation(audio_numpy.shape[0])
    audio_numpy_permute = audio_numpy[permute_idx]
    label_numpy_permute = label_numpy[permute_idx]
    audio_tensor = torch.FloatTensor(audio_numpy_permute)
    label_tensor = torch.LongTensor(label_numpy_permute)
    return audio_tensor, label_tensor.squeeze()
