import argparse
import lmdb
import pickle
import torch
import numpy as np
from torchvision.io import read_image
from torchvision.models.optical_flow import Raft_Large_Weights
from torchvision.models.optical_flow import raft_large
from tqdm import tqdm

class Data_Video:
    def __init__(self, video):
        self.frames = video.shape[0]
        self.size = video.shape[1:3]
        self.channels = video.shape[3]
        self.video = video.tobytes()

    def get_video(self):
        video = np.frombuffer(self.video, dtype = np.uint8)
        return video.reshape(self.frames, *self.size, self.channels)

class NormalImage:
    def __init__(self, weights):
        self.transforms = weights.transforms()

    def reshape_img(self, img):
        batch_size, width, height, channel = img.shape      # img shape: [batch_size, width, height, channel] 
        channel0 = img[:, :, :, 0]
        channel1 = img[:, :, :, 1]
        channel2 = img[:, :, :, 2]
        img_reshape = torch.stack([channel0, channel1, channel2], axis = 1)
        return img_reshape                                  # img shape after reshape: [batch_size, channel, width, height], this size is needed
                                                            # because of the requirement of optical flow extraction method 

    def normalize_image(self, img1_batch, img2_batch):
        img1_batch = self.reshape_img(img1_batch)
        img2_batch = self.reshape_img(img2_batch)
        return self.transforms(img1_batch, img2_batch)

def normalize_video_longer_than_clip_sec(video, frame, clip_sec):
    remainder = frame - frame // clip_sec * clip_sec
    if remainder <= clip_sec / 2:                           # remainder frame less than clip_sec/2 frames, abandon remainder
        video_complete = video[ : frame // clip_sec * clip_sec]
    else:
        frame_last = video[-1].unsqueeze(0)
        frame_repeat_num = clip_sec - remainder
        frames_repeat = frame_last.repeat(frame_repeat_num, 1, 1, 1)
        video_complete = torch.vstack([video, frames_repeat])
    return video_complete

def cal_clip_optical_flow(video_clip, image_normalizer, optic_extract_model):
    with torch.no_grad():
        video_clip_prefix = video_clip[ : -1]               # frames 0, 1, ..., clip_sec - 2
        video_clip_suffix = video_clip[1 : ]                # frames 1, 2, ..., clip_sec - 1
        video_clip_prefix_nor, video_clip_suffix_nor = image_normalizer.normalize_image(video_clip_prefix, video_clip_suffix)
        list_of_flows = optic_extract_model(video_clip_prefix_nor, video_clip_suffix_nor)
        optical_flow = list_of_flows[-1]
        optical_flow_x = optical_flow[:, 0, :, :]
        optical_flow_y = optical_flow[:, 1, :, :]
        displacement = (optical_flow_x.pow(2) + optical_flow_y.pow(2)).sqrt()
        avg_displacement = torch.sum(torch.sum(displacement, axis = -1), axis = -1) / displacement.shape[-2] / displacement.shape[-1]
        clip_avg_displacement = float(torch.mean(avg_displacement).detach().cpu())
    return clip_avg_displacement

def main():
    parser = argparse.ArgumentParser(description = "Calculate video optical flow statistics")
    parser.add_argument('--video-lmdb', default='', type = str, help = 'Raw video data lmdb')
    parser.add_argument('--frame-num-path', type = str, default='', help = 'A file containing the number of frames of the video storing in lmdb')
    parser.add_argument('--clip-duration', type = int, default = 4, help = 'Clip duration for the input video, default is 4s, i.e. 4 frames')
    parser.add_argument('--out-statistics', type = str, default='', help = 'A txt file containing the statistics of the input videos')
    parser.add_argument('--log-file', type = str, default='', help = 'A log file to record the video id which is failed for optical flow calculation')
    args = parser.parse_args()

    lmdb_video_path = args.video_lmdb
    frame_num_path = args.frame_num_path
    clip_sec = args.clip_duration
    out_statistics_file = args.out_statistics
    log_file = args.log_file

    vid2frame = {}
    with open(frame_num_path) as f_r:
        for line in f_r:
            line_strip = line.strip()
            vid, frame = line_strip.split()
            frame = int(frame)
            vid2frame[vid] = frame

    # Load optical flow extraction model
    model = raft_large(weights = Raft_Large_Weights.DEFAULT, progress = False)
    if torch.cuda.is_available():
        model = model.cuda()
    model = model.eval()

    weights = Raft_Large_Weights.DEFAULT
    image_normalizer = NormalImage(weights)
    env_video = lmdb.open(lmdb_video_path, subdir = True, readonly = True, lock = False,  meminit = False)
    with open(out_statistics_file, 'w') as f_w:
        f_w.write("# vid\toptical_flow_avg_displacement\n")
        for key in tqdm(vid2frame):
            fail_flag = 0
            with env_video.begin(write = False) as txn_video:
                video_data_bytes = txn_video.get(key.encode())
            video = pickle.loads(video_data_bytes)
            video_data = video.get_video()
            video_data_tensor = torch.from_numpy(video_data)
            frame = vid2frame[key]

            # Preprocess the video frames if the frame number cannot be divided exactly by clip_sec
            if frame <= clip_sec / 2:   # video frames less than clip_sec/2, abandon video
                fail_flag = 1
                with open(log_file, 'a') as f_log:
                    f_log.write(vid + '\t' + "too few frames" + '\n')
                    f_log.flush()
            elif clip_sec / 2 < frame < clip_sec:
                frame_last = video_data_tensor[-1].unsqueeze(0)
                frame_repeat_num = clip_sec - frame
                frames_repeat = frame_last.repeat(frame_repeat_num, 1, 1, 1)
                video_data_tensor_complete = torch.vstack([video_data_tensor, frames_repeat])
            else:
                video_data_tensor_complete = normalize_video_longer_than_clip_sec(video_data_tensor, frame, clip_sec)

            if fail_flag == 1:
                continue

            if torch.cuda.is_available():
                video_data_tensor_complete = video_data_tensor_complete.cuda()

            # Extract optical flow feature at clip level
            new_frame = video_data_tensor_complete.shape[0]
            clip_num = new_frame // clip_sec
            for i in range(clip_num):
                video_clip = video_data_tensor_complete[i * clip_sec : (i + 1) * clip_sec]
                video_clip_avg_displacement = cal_clip_optical_flow(video_clip, image_normalizer, model)
                write_line = '\t'.join([key, str(video_clip_avg_displacement)]) + '\n'
                f_w.write(write_line)
            f_w.flush()

if __name__ == '__main__':
    main()