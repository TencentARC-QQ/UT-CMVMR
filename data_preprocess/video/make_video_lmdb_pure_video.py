import sys,os,argparse
import imageio
import cv2
import pickle
import lmdb
import numpy as np
from tqdm import tqdm
from zipfile import ZipFile

class Data_Video:
    def __init__(self, video):
        self.frames = video.shape[0]
        self.size = video.shape[1:3]
        self.channels = video.shape[3]
        self.video = video.tobytes()

    def get_video(self):
        video = np.frombuffer(self.video, dtype = np.uint8)
        return video.reshape(self.frames, *self.size, self.channels)

def get_video_path(in_path_file):
    paths = []
    with open(in_path_file) as f_r:
        for path in f_r:
            paths.append(path.strip())
    return paths

def save_video_to_lmdb(video_path_list, lmdb_video_path, frame_num_path, lmdb_size, log_file, \
    resize_size = 489, max_frames = 30, extension = 'jpg'):
    video_lmdb = lmdb.open(lmdb_video_path, map_size = lmdb_size, writemap = True)
    for path in tqdm(video_path_list):
        basename = os.path.basename(path)
        basename_no_extension = os.path.splitext(basename)[0]
        img_numpy_list = []
        img_numpy = np.array([], dtype = np.uint8)
        frame_count = 0
        with ZipFile(path) as zf:
            fail_flag = 0
            try:
                for file_ in zf.namelist():
                    if not file_.endswith(extension):
                        continue
                    img_file = zf.open(file_)
                    img = imageio.imread(img_file)
                    img_divide_255 = (img / 255).astype(np.float32)     # uint8 to float32 for image resizing (need float)
                    img_divide_255 = cv2.resize(img_divide_255, (resize_size, resize_size), cv2.INTER_AREA)
                    img_uint8 = (img_divide_255 * 255).astype(np.uint8) # float32 to uint8 for storing
                    img_numpy_list.append(img_uint8)
                    frame_count += 1
                    if frame_count == max_frames:
                        break
                img_numpy = np.array(img_numpy_list)                    # [frame_num, width, height, channel]
            except:
                fail_flag = 1
                with open(log_file, 'a') as f_w:
                    f_w.write(path + '\n')
                    f_w.flush()
        if fail_flag == 1:
            continue
        else:
            with open(frame_num_path, 'a') as f_w:
                write_line = basename_no_extension + ' ' + str(frame_count)
                f_w.write(write_line + '\n')
                f_w.flush()
            with video_lmdb.begin(write = True) as txn_video:
                video = Data_Video(img_numpy)
                txn_video.put(str.encode(basename_no_extension), pickle.dumps(video))

def main():
    parser = argparse.ArgumentParser(description = "Make video lmdb database")
    parser.add_argument('--video-path', type = str, default= '', help = 'A file containing all the video paths to be processed')
    parser.add_argument('--out-path', type = str, default= '', help = 'Directory to put the video lmdb of the video')
    parser.add_argument('--frame-num-path', type = str, default= '', help = 'A file containing the frame number of the video storing in lmdb')
    parser.add_argument('--lmdb-size', type = int, default = 107374182400, help = 'Set lmdb size, default is 1GB; 0.25TB/274877906944')
    parser.add_argument('--log-file', type = str, default= '', help = 'A log file to record the file path which is failed for feature extraction')
    parser.add_argument('--resize', type = int, default = 224, help = 'Size of resized image')
    parser.add_argument('--max-frames', type = int, default = 30, help = 'Maximum input frames of each video')
    parser.add_argument('--extension', type = str, default = 'jpg', help = 'Video image file extension')
    args = parser.parse_args()

    video_path = args.video_path
    lmdb_video_path = args.out_path
    frame_num_path = args.frame_num_path
    lmdb_size = args.lmdb_size
    log_file = args.log_file
    resize_size = args.resize
    max_frames = args.max_frames
    extension = args.extension

    video_path_list = get_video_path(video_path)
    save_video_to_lmdb(video_path_list, lmdb_video_path, frame_num_path, lmdb_size, log_file, \
        resize_size = resize_size, max_frames = max_frames, extension = extension)

if __name__ == '__main__':
    main()