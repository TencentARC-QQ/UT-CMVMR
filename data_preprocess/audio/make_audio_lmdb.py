#  处理原音视频，存成lmdb文件
import sys,os,argparse
import librosa
import pickle
import lmdb
import numpy as np
from tqdm import tqdm

class Data_Audio:
    def __init__(self, audio, duration, sr):
        self.sr = sr
        self.duration = duration
        self.audio = audio.tobytes()

    def get_audio(self):
        audio = np.frombuffer(self.audio, dtype = np.float32)
        return audio

def get_audio_path(in_path_file):
    paths = []
    with open(in_path_file) as f_r:
        for path in f_r:
            paths.append(path.strip())
    return paths

def normalize_audio_longer_than_four_sec(audio, audio_sec, clip_sec, n_sample, sr):
    remainder = audio_sec - audio_sec // clip_sec * clip_sec
    if remainder <= clip_sec / 2:       # remainder duration less than clip_sec/2 seconds, abandon remainder
        audio_complete = audio[ : int(audio_sec // clip_sec * clip_sec * sr)]
    else:
        sil_add_length = int(clip_sec * sr - (n_sample - audio_sec // clip_sec * clip_sec * sr))
        audio_complete = np.hstack((audio, np.zeros(sil_add_length, dtype = np.float32)))
    return audio_complete

def save_audio_to_lmdb(audio_path_list, lmdb_audio_path, lmdb_size, log_file, sr = 16000, max_sec = 30, clip_sec = 4):
    audio_lmdb = lmdb.open(lmdb_audio_path, map_size = lmdb_size, writemap = True)
    for path in tqdm(audio_path_list):
        fail_flag = 0
        basename = os.path.basename(path)
        basename_no_extension = os.path.splitext(basename)[0]
        audio_data, sr = librosa.load(path, sr = sr)
        if len(audio_data.shape) > 1:   # multi-channel audio, need to change to single channel
            audio_data = librosa.to_mono(audio_data)
        n_sample = len(audio_data)
        audio_sec = n_sample / sr
        if audio_sec <= clip_sec / 2:   # audio duration less than clip_sec/2 seconds, abandon audio
            fail_flag = 1
            with open(log_file, 'a') as f_w:
                f_w.write(path + '\t' + "too short" + '\n')
                f_w.flush()
        elif clip_sec / 2 < audio_sec <= clip_sec:
            sil_add_length = int(clip_sec * sr - n_sample)
            audio_complete = np.hstack((audio_data, np.zeros(sil_add_length, dtype = np.float32)))
            audio_sec = int(len(audio_complete) / sr)
        else:
            if audio_sec >= max_sec:
                audio_data = audio_data[ : max_sec * sr]
                audio_sec = max_sec
            audio_complete = normalize_audio_longer_than_four_sec(audio_data, audio_sec, clip_sec, n_sample, sr)
            audio_sec = int(len(audio_complete) / sr)
        if fail_flag == 1:
            continue
        else:
            # with open(sec_num_path, 'a') as f_w:
            #     write_line = basename_no_extension + ' ' + str(audio_sec)
            #     f_w.write(write_line + '\n')
            #     f_w.flush()
            with audio_lmdb.begin(write = True) as txn_audio:
                audio = data_Audio(audio_complete, audio_sec, sr)
                txn_audio.put(str.encode(basename_no_extension), pickle.dumps(audio))

def main():
    parser = argparse.ArgumentParser(description = "Make audio lmdb database")
    parser.add_argument('--audio-path', type = str, help = 'A file containing all the audio paths to be processed')
    parser.add_argument('--out-path', type = str, help = 'Directory to put the audio lmdb of the audio')
    #  parser.add_argument('--second-num-path', type = str, help = 'A file containing the duration in seconds of the audio storing in lmdb')
    parser.add_argument('--lmdb-size', type = int, default = 1073741824, help = 'Set lmdb size, default is 1GB; 64GB/68719476736')
    parser.add_argument('--log-file', type = str, help = 'A log file to record the file path which is failed for feature extraction')
    parser.add_argument('--sample-rate', type = int, default = 16000, help = 'Set sample rate for the input audio')
    parser.add_argument('--max-duration', type = int, default = 30, help = 'Maximum duration of each input audio, default is 30s')
    parser.add_argument('--clip-duration', type = int, default = 4, help = 'Clip duration for the input audio, default is 4s')
    args = parser.parse_args()

    audio_path = args.audio_path
    lmdb_audio_path = args.out_path
    #  sec_num_path = args.second_num_path
    lmdb_size = args.lmdb_size
    log_file = args.log_file
    sr = args.sample_rate
    max_sec = args.max_duration
    clip_sec = args.clip_duration

    audio_path_list = get_audio_path(audio_path)
    save_audio_to_lmdb(audio_path_list, lmdb_audio_path, lmdb_size, log_file, \
        sr = sr, max_sec = max_sec, clip_sec = clip_sec)

if __name__ == '__main__':
    main()
