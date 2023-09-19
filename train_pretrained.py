import argparse
import logging
import os
import copy
import lmdb
import json
import pickle
from data.se_dataloader import thread_loaddata
from model_pretrained import Model_structure
from torch import optim
import torch
import torch.nn as nn
import yaml
import numpy as np
from tqdm import tqdm
import random
from datetime import datetime
from utils.recall import Recall, cal_distance
import copy
from operator import itemgetter
import glob
import warnings
warnings.filterwarnings("ignore")

class Data_Audio_feature:
    def __init__(self, audio):
        self.clip = audio.shape[0]
        self.dim = audio.shape[1]
        self.audio = audio.tobytes()

    def get_audio(self):
        audio = np.frombuffer(self.audio, dtype = np.float32)
        return audio.reshape(self.clip, self.dim)

class Data_Video_feature:
    def __init__(self, video):
        self.clip = video.shape[0]
        self.dim = video.shape[1]
        self.video = video.tobytes()

    def get_video(self):
        video = np.frombuffer(self.video, dtype = np.float32)
        return video.reshape(self.clip, self.dim)


def logging_init(path_train, path_sampler):
    file1 = logging.FileHandler(filename = path_train, mode = 'a', encoding = 'utf-8')
    fmt = logging.Formatter(fmt = "%(asctime)s - %(name)s - %(levelname)s - %(module)s: %(message)s",
        datefmt = '%Y-%m-%d %H:%M:%S')
    file1.setFormatter(fmt)

    file2 = logging.FileHandler(filename = path_sampler, mode = 'a', encoding = 'utf-8')
    file2.setFormatter(fmt)

    logger1 = logging.Logger(name = 'logger1', level = logging.INFO)
    logger2 = logging.Logger(name = 'logger2', level = logging.INFO)
    logger1.addHandler(file1)
    logger2.addHandler(file2)
    return logger1, logger2


def do_max_clip(feature, rhythm, type, max_clip=7):
    feature_complete = copy.deepcopy(feature)
    feature_rhythm = copy.deepcopy(rhythm)
    if feature.shape[0] < max_clip:
        remainder = max_clip - feature.shape[0]
        former_clip_num = feature.shape[0]
        if type == "audio":
            if remainder >= former_clip_num:
                feature_complete = np.vstack((feature_complete, np.repeat(feature, remainder // former_clip_num, 0)))
                feature_rhythm += rhythm * (remainder // former_clip_num)
            i = 0
            remainder = max_clip - feature_complete.shape[0]
            while remainder > 0:
                feature_complete = np.vstack((feature_complete, feature[i:i + 1]))
                feature_rhythm += rhythm[i:i + 1]
                remainder -= 1
                i += 1
        elif type == "video":
            feature_complete = np.vstack((feature_complete, np.repeat(feature[-1:], remainder, 0)))
            feature_rhythm += [rhythm[-1]] * remainder
    elif feature.shape[0] > max_clip:
        feature_complete = feature[: max_clip]
        feature_rhythm = rhythm[:max_clip]
    return feature_complete, feature_rhythm

# def test(model, recall_save_dir, epoch, args, lmdb_audio_path, lmdb_video_path, all_json_path, vid_list_path, aid_list_path, \
#     weight, max_clip, audio_rhythm_pkl, video_optical_json):
#     # Get the max from the topk list
#     model.eval()
#     topk = np.max(args.topk_list)
#
#     # Initiate recall
#     recall = Recall()
#     # get json dict
#     with open(all_json_path, 'r') as f:
#         pairs = json.load(f)
#
#     # Read id list
#     id_list_v_f = open(vid_list_path, 'r')
#     id_list_v = id_list_v_f.readlines()
#     id_list_a_f = open(aid_list_path, 'r')
#     id_list_a = id_list_a_f.readlines()
#     id_list_a_f.close()
#     id_list_v_f.close()
#
#     # get rhythm for audio
#     with open(audio_rhythm_pkl, 'rb') as f_r_audio:
#         audio_rhythm_dict = pickle.load(f_r_audio)
#     # get optical flow for video
#     with open(video_optical_json, 'rb') as f_r_video:
#         video_optical_dict = pickle.load(f_r_video)
#
#     # Get all the embedding of audio
#     candidate_id = []
#     candidate_value = []
#     candidate_rhythm = []
#     env_audio = lmdb.open(lmdb_audio_path, subdir = True, readonly = True, lock = False, meminit = False)
#     with env_audio.begin(write = False) as txn_audio:
#         for key in id_list_a:
#             key = key.strip('\n')
#             candidate_id.append(key)
#             audio_feature_bytes = txn_audio.get(key.encode())
#             audio_feature_bytes = pickle.loads(audio_feature_bytes)
#             audio_feature_data = audio_feature_bytes.get_audio()
#             audio_rhythm_data = audio_rhythm_dict[key]
#             audio_feature_data, audio_rhythm_data = do_max_clip(audio_feature_data, audio_rhythm_data, "audio", max_clip)
#             audio_feature_vec = torch.from_numpy(audio_feature_data)
#             audio_rhythm_vec = torch.from_numpy(np.array(audio_rhythm_data))
#             candidate_value.append(audio_feature_vec)
#             candidate_rhythm.append(audio_rhythm_vec)
#         # Reshape to be 2d
#         candidate_id = np.array(candidate_id).reshape(-1, 1)
#         candidate_value = torch.stack(candidate_value, axis = 0)
#         candidate_rhythm = torch.stack(candidate_rhythm, axis = 0)
#         # Put it into gpu
#         if torch.cuda.is_available():
#             candidate_value = candidate_value.cuda()
#             candidate_rhythm = candidate_rhythm.cuda()
#
#     # Num to count the recall
#     total_recall = np.repeat([0], len(args.topk_list))
#     env_video = lmdb.open(lmdb_video_path, subdir = True, readonly = True, lock = False, meminit = False)
#     with env_video.begin(write = False) as txn_video:
#         total_video_count = 0
#         for key in tqdm(id_list_v):
#             key = key.strip('\n')
#             aid = pairs[key][0]
#             total_video_count += 1
#             video_feature_bytes = txn_video.get(key.encode())
#             video_feature_bytes = pickle.loads(video_feature_bytes)
#             video_feature_data = video_feature_bytes.get_video()
#             video_optical_data = video_optical_dict[key]
#             video_feature_data, video_optical_data = do_max_clip(video_feature_data, video_optical_data, "video", max_clip)
#             video_feature_vec = torch.unsqueeze(torch.from_numpy(video_feature_data), axis=0)
#             video_optical_data = torch.unsqueeze(torch.from_numpy(np.array(video_optical_data)), axis=0)
#             with torch.no_grad():
#                 if torch.cuda.is_available():
#                     video_feature_vec = video_feature_vec.cuda()
#                     video_optical_data = video_optical_data.cuda()
#                 theta_a, theta_v, rho_v, rho_a = model.get_embeddings(audio=candidate_value, video=video_feature_vec, rhythm=candidate_rhythm, optical=video_optical_data)
#                 indexes, dist = recall.cal_recall(theta_a, rho_a, theta_v, rho_v, args.distance_type, topk, args.mode, weight)
#                 aid_values = np.tile(np.array([''], dtype = 'U16'), (topk, 1))
#                 for i in range(aid_values.shape[0]):
#                     aid_values[i][0] = candidate_id[indexes[i][0]][0]
#
#             # Find whether the topk cover the audio
#             for i in range(len(args.topk_list)):
#                 if (np.array(aid_values)[: args.topk_list[i]] == aid.strip('\n')).any():
#                     total_recall[i] += 1
#
#     # Make a final conclusion
#     recall_log = os.path.join(recall_save_dir, "test.log")
#     with open(recall_log, 'a+') as f_w:
#         f_w.write("test epoch: {}".format(epoch))
#         for i in range(len(args.topk_list)):
#             f_w.write("|final recall@{}: {} ".format(args.topk_list[i], total_recall[i] / len(id_list_v)))
#         f_w.write('\n')

def get_aid(candidate_aid, all_tag_id, current_tag_indexes, candidate_dict, topk, theta_v, rho_v, theta_a, rho_a, \
    recall, args, weight):
    total_count = 0      # record the total number of vecs that have been seen
    cur_topk_need = topk # record the current needed topk
    aids = []
    aids_set = set()
    candidate_labels_scanned = copy.deepcopy(all_tag_id)

    for i in range(len(current_tag_indexes)):
        tag = all_tag_id[current_tag_indexes[i]]
        if tag in candidate_dict:
            candidate_labels_scanned.remove(tag)
            cur_aids_withsame = candidate_dict[tag]
            cur_aids = []
            # remove the same aid
            for aid in cur_aids_withsame:
                if aid not in aids_set:
                    cur_aids.append(aid)
                    aids_set.add(aid)
            if len(cur_aids) == 0:  # current tag class's musics haven already been scanned
                continue
            cur_theta_a = []
            cur_rho_a = []

            for aid in cur_aids:
                cur_theta_a.append(theta_a[candidate_aid.index(aid)])
                cur_rho_a.append(rho_a[candidate_aid.index(aid)])
            cur_theta_a = torch.stack(cur_theta_a)
            cur_rho_a = torch.stack(cur_rho_a)
            cur_tag_vec_num = len(cur_aids)
            total_count += cur_tag_vec_num
            if torch.cuda.is_available():
                cur_theta_a = cur_theta_a.cuda()
                cur_rho_a = cur_rho_a.cuda()
            cur_indexes, _ = recall.cal_recall(cur_theta_a, cur_rho_a, theta_v, rho_v, args.distance_type, min(cur_tag_vec_num, cur_topk_need), args.mode, weight)
            cur_topk_need = topk - total_count
            if args.mode == "v-a":
                k = cur_indexes.shape[0]
                shortlist_aids = [cur_aids[int(cur_indexes[j][0])] for j in range(k)]
            else:
                k = cur_indexes.shape[1]
                shortlist_aids = [cur_aids[int(cur_indexes[0][j])] for j in range(k)]
            aids.extend(shortlist_aids)
            if total_count >= topk:
                break

    return aids

def extend_topk(candidate_aid, indexes, all_tag_id, t2a, topk, theta_v, rho_v, theta_a, rho_a, recall, args, weight):
    current_mid = []
    current_tag_indexes = []
    
    i = 0
    while len(current_mid) < topk:
        current_mid.extend(t2a[all_tag_id[indexes[i][0]]])
        current_mid_set_list = list(set(current_mid))
        current_mid_set_list.sort(key = current_mid.index)
        current_mid = copy.deepcopy(current_mid_set_list)
        current_tag_indexes.append(indexes[i][0])
        i += 1

    aids = get_aid(candidate_aid, all_tag_id, current_tag_indexes, t2a, topk, theta_v, rho_v, theta_a, rho_a, recall, \
        args, weight)
    return aids

# def test_tag_video(model, recall_save_dir, epoch, args, lmdb_audio_path, lmdb_video_path, lmdb_text_path, all_json_path, \
#     vid_list_path, aid_list_path, x_data_dim, y_data_dim, t_data_dim, weight):
#     # get the max from the topk list
#     model.eval()
#     topk = np.max(args.topk_list)
#
#     # Initiate recall
#     recall = Recall()
#     # Get json dict
#     with open(all_json_path, 'r') as f:
#         pairs = json.load(f)
#
#     # Read id list
#     id_list_v_f = open(vid_list_path, 'r')
#     id_list_v = id_list_v_f.readlines()
#     id_list_a_f = open(aid_list_path, 'r')
#     id_list_a = id_list_a_f.readlines()
#
#     id_list_a_f.close()
#     id_list_v_f.close()
#
#     candidate_id = []
#     candidate_value = []
#     env_audio = lmdb.open(lmdb_audio_path, subdir = True, readonly = True, lock = False, meminit = False)
#     with env_audio.begin(write = False) as txn_audio:
#         for key in id_list_a:
#             key = key.strip('\n')
#             candidate_id.append(key)
#             audio_feature_bytes = txn_audio.get(key.encode())
#             audio_feature_vec = np.frombuffer(audio_feature_bytes, np.float32).reshape(1, x_data_dim)
#             audio_feature_vec = torch.from_numpy(audio_feature_vec)
#             candidate_value.append(audio_feature_vec)
#         # Reshape to be 2d
#         candidate_value = torch.cat(candidate_value, axis = 0)
#         # Put it into gpu
#         if torch.cuda.is_available():
#             candidate_value = candidate_value.cuda()
#
#     ########## For DEBUG ##########
#     with open("candidate_aid.txt", 'w') as f_w:
#         for aid in candidate_id:
#             f_w.write(aid + '\n')
#     ########## For DEBUG ##########
#
#     # Create tagid2vid and tagid2aid dict
#     t2v = {}
#     t2a = {}
#     for vid in id_list_v:
#         vid = vid.strip("\n")
#         value = pairs[vid]
#         aid = value[0]
#         tag = value[1]
#         if tag not in t2a:
#             t2a[tag] = [aid]
#         else:
#             if aid not in t2a[tag]:
#                 t2a[tag].append(aid)
#         if tag not in t2v:
#             t2v[tag] = [vid]
#         else:
#             t2v[tag].append(vid)
#     all_tag_id = list(set(t2a.keys()))
#
#     ########## For DEBUG ##########
#     aid_to_write = []
#     for tag in t2a:
#         aid_to_write.extend(t2a[tag])
#     with open("tmp_aid_from_t2a.txt", 'w') as f_w:
#         for aid in sorted(aid_to_write):
#             f_w.write(aid + '\n')
#     ########## For DEBUG ##########
#
#     # Get rho_w for all tag
#     all_tag_value = []
#     env_text = lmdb.open(lmdb_text_path, subdir = True, readonly = True, lock = False, meminit = False)
#     with env_text.begin(write = False) as txn_text:
#         for key in all_tag_id:
#             key = key.strip('\n')
#             text_feature_bytes = txn_text.get(key.encode())
#             text_feature_vec = np.frombuffer(text_feature_bytes, np.float32).reshape(1, t_data_dim)
#             text_feature_vec = torch.from_numpy(text_feature_vec)
#             all_tag_value.append(text_feature_vec)
#         all_tag_value =  torch.concat(all_tag_value, axis = 0)
#         # Put it into gpu
#         if torch.cuda.is_available():
#             all_tag_value = all_tag_value.cuda()
#         all_tag_value = model.get_embeddings(text = all_tag_value)
#
#     # Num to count the recall
#     total_recall = np.repeat([0], len(args.topk_list))
#     env_video = lmdb.open(lmdb_video_path, subdir = True, readonly = True, lock = False, meminit = False)
#     with env_video.begin(write = False) as txn_video:
#         total_video_count = 0
#         for key in tqdm(id_list_v):
#             key = key.strip('\n')
#             aid = pairs[key][0]
#             total_video_count += 1
#             video_feature_bytes = txn_video.get(key.encode())
#             video_feature_vec = np.frombuffer(video_feature_bytes, np.float32).reshape(1, y_data_dim)
#             video_feature_vec = torch.from_numpy(video_feature_vec)
#             with torch.no_grad():
#                 if torch.cuda.is_available():
#                     video_feature_vec = video_feature_vec.cuda()
#                 theta_a, theta_v, rho_v, rho_a = model.get_embeddings(audio = candidate_value, video = video_feature_vec)
#             dist_vt = (-1) * cal_distance(all_tag_value, rho_v, args.distance_type, args.mode)
#             #  from biggest to the smallest
#             indexes = torch.argsort(dist_vt, dim = 0, descending = True)
#             aids = extend_topk(candidate_id, indexes, all_tag_id, t2a, topk, theta_v, rho_v, theta_a, rho_a, recall, args, weight)
#             # find whether the topk cover the audio
#             for i in range(len(args.topk_list)):
#                 if (np.array(aids)[: args.topk_list[i]] == aid.strip('\n')).any():
#                     total_recall[i] += 1
#
#             # make a final conclusion
#         recall_log = os.path.join(recall_save_dir, "test_tag_video.log")
#         with open(recall_log, 'a+') as f_w:
#             f_w.write("test epoch with tag: {}".format(epoch))
#             for i in range(len(args.topk_list)):
#                 f_w.write("|final recall@{}: {} ".format(args.topk_list[i], total_recall[i] / len(id_list_v)))
#             f_w.write('\n')

def test_tag_tag(model, recall_save_dir, epoch, args, lmdb_audio_path, lmdb_video_path, lmdb_text_path, all_json_path, \
    vid_list_path, aid_list_path, t_data_dim, weight, max_clip, audio_rhythm_pkl, video_optical_json):
    # get the max from the topk list
    model.eval()
    topk = np.max(args.topk_list)

    # Initiate recall
    recall = Recall()
    # Get json dict
    with open(all_json_path, 'r') as f:
        pairs = json.load(f)

    # Read id list
    id_list_v_f = open(vid_list_path, 'r')
    id_list_v = id_list_v_f.readlines()
    id_list_a_f = open(aid_list_path, 'r')
    id_list_a = id_list_a_f.readlines()

    # get rhythm for audio
    with open(audio_rhythm_pkl, 'rb') as f_r_audio:
        audio_rhythm_dict = pickle.load(f_r_audio)
    # get optical flow for video
    with open(video_optical_json, 'rb') as f_r_video:
        video_optical_dict = pickle.load(f_r_video)

    id_list_a_f.close()
    id_list_v_f.close()

    candidate_id = []
    candidate_value = []
    candidate_rhythm = []
    env_audio = lmdb.open(lmdb_audio_path, subdir = True, readonly = True, lock = False, meminit = False)
    with env_audio.begin(write = False) as txn_audio:
        for key in id_list_a:
            key = key.strip('\n')
            candidate_id.append(key)
            audio_feature_bytes = txn_audio.get(key.encode())
            audio_feature_bytes = pickle.loads(audio_feature_bytes)
            audio_feature_data = audio_feature_bytes.get_audio()
            audio_rhythm_data = audio_rhythm_dict[key]
            audio_feature_data, audio_rhythm_data = do_max_clip(audio_feature_data, audio_rhythm_data, "audio", max_clip)
            audio_feature_vec = torch.from_numpy(audio_feature_data)
            audio_rhythm_vec = torch.from_numpy(np.array(audio_rhythm_data))
            candidate_value.append(audio_feature_vec)
            candidate_rhythm.append(audio_rhythm_vec)
        # Reshape to be 2d
        candidate_value = torch.stack(candidate_value, axis = 0)
        candidate_rhythm = torch.stack(candidate_rhythm, axis = 0)
        # Put it into gpu
        if torch.cuda.is_available():
            candidate_value = candidate_value.cuda()
            candidate_rhythm = candidate_rhythm.cuda()
    # Create tagid2vid and tagid2aid dict
    t2v = {}
    t2a = {}
    for vid in id_list_v:
        vid = vid.strip("\n")
        value = pairs[vid]
        aid = value[0]
        tag = value[1]
        if tag not in t2a:
            t2a[tag] = [aid]
        else:
            if aid not in t2a[tag]:
                t2a[tag].append(aid)
        if tag not in t2v:
            t2v[tag] = [vid]
        else:
            t2v[tag].append(vid)
    all_tag_id = list(set(t2a.keys()))

    # Get w for all tag
    all_tag_value = []
    env_text = lmdb.open(lmdb_text_path, subdir = True, readonly = True, lock = False, meminit = False)
    with env_text.begin(write = False) as txn_text:
        for key in all_tag_id:
            key = key.strip('\n')
            text_feature_bytes = txn_text.get(key.encode())
            text_feature_vec = np.frombuffer(text_feature_bytes, np.float32).reshape(1, t_data_dim)
            text_feature_vec = torch.from_numpy(text_feature_vec)
            all_tag_value.append(text_feature_vec)
        all_tag_value =  torch.concat(all_tag_value, axis = 0)
        # Put it into gpu
        if torch.cuda.is_available():
            all_tag_value = all_tag_value.cuda()

    # Num to count the recall
    total_recall = np.repeat([0], len(args.topk_list))
    env_video = lmdb.open(lmdb_video_path, subdir = True, readonly = True, lock = False, meminit = False)
    with env_video.begin(write = False) as txn_video:
        total_video_count = 0
        for key in tqdm(id_list_v):
            key = key.strip('\n')
            aid = pairs[key][0]
            tag = pairs[key][1]
            total_video_count += 1
            video_feature_bytes = txn_video.get(key.encode())
            video_feature_bytes = pickle.loads(video_feature_bytes)
            video_feature_data = video_feature_bytes.get_video()
            video_optical_data = video_optical_dict[key]
            video_feature_data, video_optical_data = do_max_clip(video_feature_data, video_optical_data, "video", max_clip)
            video_feature_vec = torch.unsqueeze(torch.from_numpy(video_feature_data), axis=0)
            video_optical_vec = torch.unsqueeze(torch.from_numpy(np.array(video_optical_data)), axis=0)
            # get the tag embedding for the video
            with env_text.begin(write=False) as txn_text:
                cur_text_embed = txn_text.get(tag.encode())
                text_feature_vec = np.frombuffer(cur_text_embed, np.float32).reshape(1, t_data_dim)
                text_feature_vec = torch.from_numpy(text_feature_vec)
                if torch.cuda.is_available():
                    text_feature_vec = text_feature_vec.cuda()

            with torch.no_grad():
                if torch.cuda.is_available():
                    video_feature_vec = video_feature_vec.cuda()
                    video_optical_vec = video_optical_vec.cuda()
                theta_a, theta_v, rho_v, rho_a = model.get_embeddings(audio = candidate_value, video = video_feature_vec, rhythm=candidate_rhythm, optical=video_optical_vec)
            dist_tt = (-1) * cal_distance(all_tag_value, text_feature_vec, args.distance_type, args.mode)
            #  from biggest to the smallest
            indexes = torch.argsort(dist_tt, dim = 0, descending = True)
            aids = extend_topk(candidate_id, indexes, all_tag_id, t2a, topk, theta_v, rho_v, theta_a, rho_a, recall, args, weight)
            # find whether the topk cover the audio
            for i in range(len(args.topk_list)):
                if (np.array(aids)[: args.topk_list[i]] == aid.strip('\n')).any():
                    total_recall[i] += 1

            # make a final conclusion
        recall_log = os.path.join(recall_save_dir, "test_tag_tag.log")
        with open(recall_log, 'a+') as f_w:
            f_w.write("test epoch with embedding: {}".format(epoch))
            for i in range(len(args.topk_list)):
                f_w.write("|final recall@{}: {} ".format(args.topk_list[i], total_recall[i] / len(id_list_v)))
            f_w.write('\n')

# def test_dam(model, recall_save_dir, epoch, args, lmdb_audio_path, lmdb_video_path, all_json_path, \
#     vid_list_path, aid_list_path, x_data_dim, y_data_dim, weight):
#     # get the max from the topk list
#     model.eval()
#     topk = np.max(args.topk_list)
#
#     # Initiate recall
#     recall = Recall()
#     # Get json dict
#     with open(all_json_path, 'r') as f:
#         pairs = json.load(f)
#
#     # Read id list
#     id_list_v_f = open(vid_list_path, 'r')
#     id_list_v = id_list_v_f.readlines()
#     id_list_a_f = open(aid_list_path, 'r')
#     id_list_a = id_list_a_f.readlines()
#
#     id_list_a_f.close()
#     id_list_v_f.close()
#
#     candidate_id = []
#     candidate_value = []
#     env_audio = lmdb.open(lmdb_audio_path, subdir = True, readonly = True, lock = False, meminit = False)
#     with env_audio.begin(write = False) as txn_audio:
#         for key in id_list_a:
#             key = key.strip('\n')
#             candidate_id.append(key)
#             audio_feature_bytes = txn_audio.get(key.encode())
#             audio_feature_vec = np.frombuffer(audio_feature_bytes, np.float32).reshape(1, x_data_dim)
#             audio_feature_vec = torch.from_numpy(audio_feature_vec)
#             candidate_value.append(audio_feature_vec)
#         # Reshape to be 2d
#         candidate_value = torch.cat(candidate_value, axis = 0)
#         # Put it into gpu
#         if torch.cuda.is_available():
#             candidate_value = candidate_value.cuda()
#
#     # Create tag_id2vid and tag_id2aid dict
#     t2v = {}
#     t2a = {}
#     all_tag_id = []
#     all_tag_label = []
#     for vid in id_list_v:
#         vid = vid.strip("\n")
#         value = pairs[vid]
#         aid = value[0]
#         tag = value[1]
#         label = int(value[2])
#         if tag not in all_tag_id:
#             all_tag_id.append(tag)
#             all_tag_label.append(label)
#         if tag not in t2a:
#             t2a[tag] = [aid]
#         else:
#             if aid not in t2a[tag]:
#                 t2a[tag].append(aid)
#         if tag not in t2v:
#             t2v[tag] = [vid]
#         else:
#             t2v[tag].append(vid)
#
#     # Num to count the recall
#     total_recall = np.repeat([0], len(args.topk_list))
#     env_video = lmdb.open(lmdb_video_path, subdir = True, readonly = True, lock = False, meminit = False)
#     with env_video.begin(write = False) as txn_video:
#         total_video_count = 0
#         for key in tqdm(id_list_v):
#             key = key.strip('\n')
#             aid = pairs[key][0]
#             tag = pairs[key][1]
#             total_video_count += 1
#             video_feature_bytes = txn_video.get(key.encode())
#             video_feature_vec = np.frombuffer(video_feature_bytes, np.float32).reshape(1, y_data_dim)
#             video_feature_vec = torch.from_numpy(video_feature_vec)
#
#             with torch.no_grad():
#                 if torch.cuda.is_available():
#                     video_feature_vec = video_feature_vec.cuda()
#                 theta_a, theta_v, rho_v, rho_a = model.get_embeddings(audio = candidate_value, video = video_feature_vec)
#                 tile_rho_v = torch.tile(rho_v, (rho_a.shape[0], 1))
#                 # get dam output
#                 costh = model.damsoftmax_loss(x_embed = rho_a, y_embed = tile_rho_v, if_testing = True)
#                 # remove
#                 costh = costh[:, all_tag_label]
#                 max_per_line, index_per_line = torch.max(costh, 1)
#                 _, index = torch.max(max_per_line, -1)
#                 top1_outputs = costh[index]
#                 # from biggest to the smallest
#                 indexes = torch.argsort(top1_outputs, dim = 0, descending = True)
#                 indexes = indexes.reshape(indexes.shape[0], -1)
#                 aids = extend_topk(candidate_id, indexes, all_tag_id, t2a, topk, theta_v, rho_v, theta_a, rho_a, recall, args, weight)
#                 # find whether the topk cover the audio
#                 for i in range(len(args.topk_list)):
#                     if (np.array(aids)[: args.topk_list[i]] == aid.strip('\n')).any():
#                         total_recall[i] += 1
#
#             # make a final conclusion
#         recall_log = os.path.join(recall_save_dir, "test_dam.log")
#         with open(recall_log, 'a+') as f_w:
#             f_w.write("test epoch with damloss: {}".format(epoch))
#             for i in range(len(args.topk_list)):
#                 f_w.write("|final recall@{}: {} ".format(args.topk_list[i], total_recall[i] / len(id_list_v)))
#             f_w.write('\n')

def main():
    # Input parameters for train
    parser = argparse.ArgumentParser(description = "To train the AVCA model, a configuration .yaml file is needed")
    parser.add_argument('--yaml-path', type = str, default = "./config/config_pre_rhythm.yaml", \
        help = 'a yaml file to store important parameters')
    parser.add_argument('--finetune-model', type = str, help = 'a trained model for finetuning')
    # Input parameters for test
    parser.add_argument('--mode', type = str, default = "v-a", choices = ["v-a", "a-v"], help = "decide the matching mode")
    parser.add_argument('--distance-type', type = str, default = "COS", choices = ["L2", "COS"], \
        help = "method to calculate the distance")
    parser.add_argument("--topk-list", type = list, default = [1, 5, 10, 12, 25], help = "list of topk")
    parser.add_argument("--text_embed_mode", type = str, default = "1", help = "choose the type of text embedding")
    parser.add_argument("--resume-date", type=str, default="20221025", help="choose the type of text embedding")
    args = parser.parse_args()

    # Read config_proto.yaml
    with open(args.yaml_path, 'r', encoding = 'utf-8') as f_r:
        parameters = yaml.load(f_r.read(), Loader = yaml.Loader)

    # Related to structure of model
    a_data_dim = parameters["a_data_dim"]
    v_data_dim = parameters["v_data_dim"]
    t_data_dim = parameters["t_data_dim"]
    init_method = parameters["init_method"]
    dim_rho = parameters["dim_rho"]
    dim_theta = parameters["dim_theta"]
    encoder_hidden_size = parameters["encoder_hidden_size"]
    decoder_hidden_size = parameters["decoder_hidden_size"]
    dropout_encoder = parameters["dropout_encoder"]
    dropout_decoder = parameters["dropout_decoder"]
    depth_transformer = parameters["depth_transformer"]
    additional_dropout = parameters["additional_dropout"]
    momentum = parameters["momentum"]
    first_additional_triplet = parameters["first_additional_triplet"]
    second_additional_triplet = parameters["second_additional_triplet"]
    margin = parameters["margin"]
    max_clip = parameters["max_clip"]
    a_enc_dmodel = parameters["a_enc_dmodel"]
    v_enc_dmodel = parameters["v_enc_dmodel"]
    nlayers = parameters["nlayers"]
    dropout_enc = parameters["dropout_enc"]
    enc_hidden_dim = parameters["enc_hidden_dim"]
    cross_hidden_dim = parameters["cross_hidden_dim"]
    inter_margin = parameters["inter_margin"]
    inter_topk = parameters["inter_topk"]
    intra_topk = parameters["intra_topk"]
    add_rhythm = parameters["add_rhythm"]

    # loss feature
    keep_prob = parameters["keep_prob"]
    loss_mask = parameters["loss_mask"]

    # Training control parameters
    batch_size = parameters["batch_size"]
    epochs = parameters["epochs"]
    original_learning_rate = parameters["original_learning_rate"]
    max_grad_norm = parameters["max_grad_norm"]
    decay_rate = parameters["decay_rate"]

    # test parameters
    weight = parameters["weight"]

    # Necessary input file paths
    lmdb_audio_path = parameters["lmdb_audio_path"]
    lmdb_video_path = parameters["lmdb_video_path"]
    paraphrase_lmdb_text_path = parameters["paraphrase_lmdb_text_path"]
    all_lmdb_text_path = parameters["all_lmdb_text_path"]
    train_txt_vid_list = parameters["train_txt_vid_list"]
    test_txt_vid_list = parameters["test_txt_vid_list"]
    valid_txt_vid_list = parameters["valid_txt_vid_list"]
    json_path = parameters["json_path"]
    save_dir = parameters["save_dir"]
    test_txt_aid_list = parameters["test_txt_aid_list"]
    audio_rhythm_pkl = parameters["audio_rhythm_pkl"]
    video_optical_json = parameters["video_optical_json"]

    # set the special dates to be resumed
    if args.resume_date != "":
        datetime_ = args.resume_date
    else:
        datetime_ = datetime.today().strftime('%Y%m%d')

    save_dir = save_dir if save_dir.endswith('/') else save_dir + '/'
    save_dir = save_dir + "new_{}_weight{}_batch{}_init-{}_dimrho{}_dimtheta{}_ehsize{}_dhsize{}_de{}_dd{}_depth_trans{}_ad{}_margin{}_intmar{}_lossmask{}_nlayer{}_enchid{}_crosshid{}_adim{}_vdim{}_adrhm{}/"\
        .format(datetime_, weight,\
            batch_size, init_method, dim_rho, dim_theta, encoder_hidden_size, decoder_hidden_size, \
            dropout_encoder, dropout_decoder, depth_transformer, additional_dropout, \
            margin, inter_margin, ''.join([str(i) for i in loss_mask]), nlayers, enc_hidden_dim, cross_hidden_dim, a_data_dim, v_data_dim, add_rhythm)


    # get all tag of the data
    f = open(json_path)
    all_pair = json.load(f)
    f.close()
    all_tag_id = []
    for vid, value in all_pair.items():
        tag = value[1]
        if tag not in all_tag_id:
            all_tag_id.append(tag)
    target_dim = len(all_tag_id)

    # Instantiate the model
    model = Model_structure(a_data_dim, v_data_dim, t_data_dim, dim_theta, dim_rho, init_method, \
        encoder_hidden_size, decoder_hidden_size, depth_transformer, additional_dropout, momentum, \
        first_additional_triplet, second_additional_triplet, dropout_encoder, dropout_decoder, \
        margin, original_learning_rate, target_dim, keep_prob, loss_mask, max_grad_norm, decay_rate, \
        inter_margin, inter_topk, intra_topk, batch_size, args.distance_type, a_enc_dmodel, v_enc_dmodel, \
                            nlayers, dropout_enc, enc_hidden_dim, cross_hidden_dim, max_clip, add_rhythm)


    # check if exists terminated models
    scheduler_state = None
    optimizer_state = None
    start_epoch = 0
    total_train_step = 0
    file_list = glob.glob(os.path.join(save_dir, "*.pth"))
    file_list = [file.split("/")[-1] for file in file_list]
    if len(file_list) != 0:
        if os.path.exists(save_dir + "epoch_lr_step.pkl"):
            with open(save_dir + "epoch_lr_step.pkl", 'rb') as fo:  # load pkl file, containing lr/step info
                dict_data = pickle.load(fo, encoding = 'bytes')
                total_train_step = dict_data["step"]
                scheduler_state = dict_data["scheduler"]
                optimizer_state = dict_data["optimizer"]
        cur_pth = ''
        cur_max_epoch = 0
        for file in file_list:
            name = os.path.splitext(file)[0]
            epoch = int(name.split("_")[2])
            if epoch > cur_max_epoch:
                cur_max_epoch = epoch
                cur_pth = file
        if not torch.cuda.is_available():
            checkpoint = torch.load(os.path.join(save_dir, cur_pth), map_location = "cpu")
        else:
            checkpoint = torch.load(os.path.join(save_dir, cur_pth))
        model.load_state_dict(checkpoint)
        print('Successfully reloaded epoch {} model! Training restart!'.format(cur_max_epoch))
        start_epoch = cur_max_epoch
    else:
        print('No trained model found! Training from scratch!')

    train_log_file = save_dir + "train.log"
    sampler_log_file = save_dir + "sampler.log"

    # Make the checkpoints dir
    os.makedirs(save_dir, exist_ok=True)

    # Initialize the log file
    logger1, logger2 = logging_init(train_log_file, sampler_log_file)

    # Initialize random seed if exists
    try:
        seed = parameters["seed"]
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
    except:
        pass

    # Finetune model if a trained model is given
    if args.finetune_model:
        if not torch.cuda.is_available():
            checkpoint = torch.load(args.finetune_model, map_location = "cpu")
        else:
            checkpoint = torch.load(args.finetune_model)
        model.load_state_dict(checkpoint)

    # Check GPU availability
    if torch.cuda.is_available():
        model.cuda()

    # Text embedding model selection
    if args.text_embed_mode == "1":
        text_path = paraphrase_lmdb_text_path
    else:
        text_path = all_lmdb_text_path

    # Get loader for train and valid
    train_loader = thread_loaddata(lmdb_audio_path, lmdb_video_path, text_path, batch_size,
        train_txt_vid_list, json_path, logger2, "train", audio_rhythm_pkl, video_optical_json, max_clip)
    valid_loader = thread_loaddata(lmdb_audio_path, lmdb_video_path, text_path, batch_size,
        valid_txt_vid_list, json_path, logger2, "val", audio_rhythm_pkl, video_optical_json, max_clip)

    # Start train
    for i in range(start_epoch, epochs):
        print("---------------Epoch {} training start--------------".format(i + 1))
        # Start to train
        model.train()
        train_loss_total = []
        train_ce_acc_list = []
        # train_dam_acc_list = []
        # train_adv_acc_list = []

        for step, (p_audio_tensor, p_video_tensor, n_audio_tensor, n_video_tensor, p_aid_numpy, p_vid_numpy, n_aid_numpy,
            n_vid_numpy, p_target_numpy, n_target_numpy, p_text_tensor, n_text_tensor, p_label, n_label, p_rhythm, p_optical, n_rhythm, n_optical) in tqdm(enumerate(train_loader)):
            if torch.cuda.is_available():
                p_audio_tensor = p_audio_tensor.cuda()
                p_video_tensor = p_video_tensor.cuda()
                n_audio_tensor = n_audio_tensor.cuda()
                n_video_tensor = n_video_tensor.cuda()
                p_text_tensor = p_text_tensor.cuda()
                n_text_tensor = n_text_tensor.cuda()
                p_label = p_label.cuda()
                n_label = n_label.cuda()
                p_rhythm = p_rhythm.cuda()
                p_optical = p_optical.cuda()
                n_rhythm = n_rhythm.cuda()
                n_optical = n_optical.cuda()
            # Count the steps
            total_train_step = total_train_step + 1

            # Forward and backward
            loss, _, ce_acc = model.optimize_params(p_audio_tensor, p_video_tensor, p_text_tensor, \
                n_audio_tensor, n_video_tensor, n_text_tensor, p_label, p_aid_numpy, p_vid_numpy, p_rhythm, p_optical, n_rhythm, n_optical, \
                scheduler_state = scheduler_state, optimizer_state = optimizer_state, optimize = True, if_testing = False)
            scheduler_state = None
            optimizer_state = None
            # Update the learning rate
            if (total_train_step % 1000) == 0:
                model.optimize_scheduler()
            train_loss_total.append(loss.item())
            train_ce_acc_list.append(ce_acc.item())
            # train_dam_acc_list.append(dam_acc.item())
            # train_adv_acc_list.append(adv_acc.item())

        # Validation step per epoch, schedule the learning rate
        print("-------------Epoch {} validation start--------------".format(i + 1))
        val_loss_total = []
        val_ce_acc_list = []
        # val_dam_acc_list = []
        # val_adv_acc_list = []
        with torch.no_grad():
            for step, (p_audio_tensor, p_video_tensor, n_audio_tensor, n_video_tensor, p_aid_numpy, p_vid_numpy, n_aid_numpy,
                n_vid_numpy, p_target_numpy, n_target_numpy, p_text_tensor, n_text_tensor, p_label, n_label, p_rhythm, p_optical, n_rhythm, n_optical) in tqdm(enumerate(valid_loader)):
                if torch.cuda.is_available():
                    p_audio_tensor = p_audio_tensor.cuda()
                    p_video_tensor = p_video_tensor.cuda()
                    n_audio_tensor = n_audio_tensor.cuda()
                    n_video_tensor = n_video_tensor.cuda()
                    p_text_tensor = p_text_tensor.cuda()
                    n_text_tensor = n_text_tensor.cuda()
                    p_label = p_label.cuda()
                    n_label = n_label.cuda()
                    p_optical = p_optical.cuda()
                    p_rhythm = p_rhythm.cuda()
                    n_optical = n_optical.cuda()
                    n_rhythm = n_rhythm.cuda()

                # Forward only
                loss, _, ce_acc = model.optimize_params(p_audio_tensor, p_video_tensor, p_text_tensor, \
                    n_audio_tensor, n_video_tensor, n_text_tensor, p_label, p_aid_numpy, p_vid_numpy, p_rhythm, p_optical, n_rhythm, n_optical, optimize = False, if_testing = False)
                val_loss_total.append(loss.item())
                val_ce_acc_list.append(ce_acc.item())
                # val_dam_acc_list.append(dam_acc.item())
                # val_adv_acc_list.append(adv_acc.item())

        # Begin test for each epoch
        # test(model, save_dir, i + 1, args, lmdb_audio_path, lmdb_video_path, json_path, \
        #     test_txt_vid_list, test_txt_aid_list, weight, max_clip, audio_rhythm_pkl, video_optical_json)
        # test_tag_video(model, save_dir, i + 1, args, lmdb_audio_path, lmdb_video_path, text_path, json_path, \
        #     test_txt_vid_list, test_txt_aid_list, a_data_dim, v_data_dim, t_data_dim, weight)
        test_tag_tag(model, save_dir, i + 1, args, lmdb_audio_path, lmdb_video_path, text_path, json_path, \
            test_txt_vid_list, test_txt_aid_list, t_data_dim, weight, max_clip, audio_rhythm_pkl, video_optical_json)
        # test_dam(model, save_dir, i + 1, args, lmdb_audio_path, lmdb_video_path, json_path, \
        #     test_txt_vid_list, test_txt_aid_list, a_data_dim, v_data_dim, weight)

        # Record the current epoch info
        current_lr = model.optimizer.param_groups[0]['lr']
        print("Epoch %d learning rate: %4g" % (i + 1, current_lr))
        logger1.info(
            "| INFO | Epochs : {}| Train Loss: {:.4f} | Valid Loss: {:.4f} | Train Ce Acc: {:.4%} | Valid Ce Acc: {:.4%} |" \
            "Learning Rate: {:.4g}".format(
                i + 1, \
                float(np.mean(train_loss_total)), float(np.mean(val_loss_total)), float(np.mean(train_ce_acc_list)),
                float(np.mean(val_ce_acc_list)), \
                float(current_lr)))

        # choose to save all the model dict
        torch.save(model.state_dict(), save_dir + 'model_epoch_{}_valid_loss_{:.4f}_step_{}.pth'. \
            format(i + 1, np.mean(val_loss_total), total_train_step))
        # save epoch、scheduler、optimizer、total_train_step，rewrite
        states = {
            'epoch': i,
            'scheduler': model.scheduler.state_dict(),
            'optimizer': model.optimizer.state_dict(),
            'step': total_train_step
        }

        with open(save_dir + "epoch_lr_step.pkl", 'wb') as fo:
            pickle.dump(states, fo)

if __name__ == '__main__':
    main()
