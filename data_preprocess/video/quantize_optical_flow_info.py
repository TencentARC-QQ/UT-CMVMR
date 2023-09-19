# flake8: noqa
import sys,os,argparse,pickle
import numpy as np
import intervaltree
from ..audio.quantize_rhythm_info import get_intervaltree



def main():
    parser = argparse.ArgumentParser(description = "Quantize optical flow information")
    parser.add_argument('--optic-flow-statistics', type = str, default='', help = 'Optical flow statistics txt file')
    parser.add_argument('--quantize-class-dim', type = int, default = 65, help = 'Representing the class number of the clip-level \
        per pixel average optical flow displacement (default is slicing 0-320 into 64 intervals)')
    parser.add_argument('--max-displacement', type = int, default = 320, help = 'Representing the maximum optical flow displacement \
        of a pixel')
    parser.add_argument('--out-quantize-file', type = str, default='', help = 'A pickle file including the quantized optical flow information of \
        each video in clip style')
    args = parser.parse_args()

    optic_flow_statistics = args.optic_flow_statistics
    quantize_class_dim = args.quantize_class_dim
    max_displacement = args.max_displacement
    out_quantize_file = args.out_quantize_file

    tree, interval2index = get_intervaltree(max_displacement if max_displacement else 320 , quantize_class_dim if quantize_class_dim else 65)

    vid2statistics = {}
    with open(optic_flow_statistics) as f_r:
        f_r.readline()
        for line in f_r:
            line_strip = line.strip()
            terms = line_strip.split()
            vid, avg_clip_displacement = terms
            avg_clip_displacement = float(avg_clip_displacement)
            interval = list(tree[avg_clip_displacement])[0]
            interval_idx = interval2index[interval]
            if vid not in vid2statistics:
                vid2statistics[vid] = [interval_idx]
            else:
                vid2statistics[vid].append(interval_idx)

    with open(out_quantize_file, 'wb') as f_w:
        pickle.dump(vid2statistics, f_w)

if __name__ == '__main__':
    main()