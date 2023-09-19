import sys,os,argparse,pickle
import numpy as np
import intervaltree

def get_intervaltree(clip_sec = 4, quantize_class_dim = 65):
    points = np.linspace(0, clip_sec, quantize_class_dim)
    intervals = [[points[i], points[i + 1]] for i in range(len(points) - 1)]    # (quantize_class_dim - 1) intervals
    tree = intervaltree.IntervalTree()
    for interval in intervals:
        tree[interval[0] : interval[1]] = (interval[0], interval[1])
    interval2index = {}
    sorted_tree = sorted(tree)
    for i in range(len(sorted_tree)):
        interval2index[sorted_tree[i]] = i
    return tree, interval2index

def main():
    parser = argparse.ArgumentParser(description = "Quantize rhythm information")
    parser.add_argument('--rhythm-statistics', type = str, help = 'Rhythm statistics txt file')
    parser.add_argument('--clip-duration', type = int, default = 4, help = 'Clip duration for the input audio, default is 4s')
    parser.add_argument('--quantize-class-dim', type = int, default = 65, help = 'Representing the class number of the average \
        interval width between two adjacent beats in a clip (default is slicing 0-4s into 64 intervals, if clip beat number is 0/1, \
        the class index will be 64, which is the 65th class)')
    parser.add_argument('--out-quantize-file', type = str, help = 'A pickle file including the quantized rhythm information of each \
        music in clip style')
    args = parser.parse_args()

    rhythm_statistics = args.rhythm_statistics
    clip_sec = args.clip_duration
    quantize_class_dim = args.quantize_class_dim
    out_quantize_file = args.out_quantize_file

    tree, interval2index = get_intervaltree(clip_sec, quantize_class_dim)

    aid2statistics = {}
    with open(rhythm_statistics) as f_r:
        f_r.readline()
        for line in f_r:
            line_strip = line.strip()
            terms = line_strip.split()
            aid, n_beats, avg_beat_strength, avg_beat_interval_width = terms
            n_beats = int(n_beats)
            if n_beats == 0:
                if aid not in aid2statistics:
                    aid2statistics[aid] = [[n_beats, 0, quantize_class_dim - 1]]
                else:
                    aid2statistics[aid].append([n_beats, 0, quantize_class_dim - 1])
            elif n_beats == 1:
                avg_beat_strength = int(round(float(avg_beat_strength)))
                if aid not in aid2statistics:
                    aid2statistics[aid] = [[n_beats, avg_beat_strength, quantize_class_dim - 1]]
                else:
                    aid2statistics[aid].append([n_beats, avg_beat_strength, quantize_class_dim - 1])
            else:
                avg_beat_strength = int(round(float(avg_beat_strength)))
                avg_beat_interval_width = float(avg_beat_interval_width)
                interval = list(tree[avg_beat_interval_width])[0]
                interval_idx = interval2index[interval]
                if aid not in aid2statistics:
                    aid2statistics[aid] = [[n_beats, avg_beat_strength, interval_idx]]
                else:
                    aid2statistics[aid].append([n_beats, avg_beat_strength, interval_idx])

    with open(out_quantize_file, 'wb') as f_w:
        pickle.dump(aid2statistics, f_w)

if __name__ == '__main__':
    main()