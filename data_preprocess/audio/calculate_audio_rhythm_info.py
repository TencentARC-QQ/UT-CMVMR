import sys,os,argparse,pickle
import lmdb
import librosa
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

def cal_rhythm_info(audio_clip, sr):
    onset_strength = librosa.onset.onset_strength(y = audio_clip, sr = sr)
    _, beats = librosa.beat.beat_track(onset_envelope = onset_strength, sr = sr)
    clip_n_beats = len(beats)
    beat_strength = onset_strength[beats]
    clip_beat_avg_strength = np.mean(beat_strength)
    beat_times = librosa.frames_to_time(beats, sr = sr)
    beat_diff_times = np.diff(beat_times)
    clip_avg_beat_interval_time = np.mean(beat_diff_times)

    return clip_n_beats, clip_beat_avg_strength, clip_avg_beat_interval_time

def main():
    parser = argparse.ArgumentParser(description = "Calculate audio rhythm information and statistics")
    parser.add_argument('--audio-lmdb', type = str, help = 'Raw audio data lmdb')
    parser.add_argument('--second-num-path', type = str, help = 'A file containing the duration in seconds of the audio storing in lmdb')
    parser.add_argument('--clip-duration', type = int, default = 4, help = 'Clip duration for the input audio, default is 4s')
    parser.add_argument('--out-statistics', type = str, help = 'A txt file containing the statistics of the input audios')
    args = parser.parse_args()

    lmdb_audio_path = args.audio_lmdb
    sec_num_path = args.second_num_path
    clip_sec = args.clip_duration
    out_statistics_file = args.out_statistics

    aid2dur = {}
    with open(sec_num_path) as f_r:
        for line in f_r:
            line_strip = line.strip()
            aid, dur = line_strip.split()
            dur = int(dur)
            aid2dur[aid] = dur

    key2statistics = {}
    env_audio = lmdb.open(lmdb_audio_path, subdir = True, readonly = True, lock = False,  meminit = False)
    with open(out_statistics_file, 'w') as f_w:
        f_w.write("# aid\tn_beat\tbeat_strength\tbeat_interval\n")
        for key in tqdm(aid2dur):
            with env_audio.begin(write = False) as txn_audio:
                audio_data_bytes = txn_audio.get(key.encode())
            audio = pickle.loads(audio_data_bytes)
            audio_data = audio.get_audio()
            sr = int(audio.sr)
            duration = aid2dur[key]
            clip_num = duration // clip_sec
            for i in range(clip_num):
                waveform = audio_data[i * sr * clip_sec : (i + 1) * sr * clip_sec]
                clip_n_beats, clip_beat_avg_strength, clip_avg_beat_interval_time = cal_rhythm_info(waveform, sr)
                write_line = '\t'.join([key, str(clip_n_beats), str(clip_beat_avg_strength), str(clip_avg_beat_interval_time)]) + '\n'
                f_w.write(write_line)
            f_w.flush()

if __name__ == '__main__':
    main()