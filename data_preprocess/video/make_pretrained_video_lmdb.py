import lmdb, argparse, torch, yaml, os, pickle, cv2, sys, json
sys.path.append(os.path.abspath(r'/home/conformer_pretrained_video'))
from ..model_Conformer import Conformer
import numpy as np
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

class Data_Video_feature:
    def __init__(self, video):
        self.clip = video.shape[0]
        self.dim = video.shape[1]
        self.video = video.tobytes()

    def get_video(self):
        video = np.frombuffer(self.video, dtype = np.float32)
        return video.reshape(self.clip, self.dim)

def get_video_lst(in_path_file):
    aids = []
    with open(in_path_file) as f_r:
        for path in f_r:
            path = path.strip()
            # basename = os.path.basename(path)
            # basename_no_extension = os.path.splitext(basename)[0]
            # aids.append(basename_no_extension)
            aids.append(path)
    return aids

def get_feature(model, vid, lmdb_from_path, clip_sec, seq_length, max_clip_num, less_twof_vid_path):
    def normalize_video_longer_than_four_sec(video, video_sec, clip_sec):
        remainder = video_sec - video_sec // clip_sec * clip_sec  # check the number of frames which could not be put into a complete clip
        if remainder <= clip_sec / 2:  # remainder duration less than clip_sec/2 seconds, abandon remainder
            video_complete = video[: int(video_sec // clip_sec * clip_sec)]
        else:
            video_add_length = (video_sec // clip_sec + 1) * clip_sec - video_sec
            video_complete = np.vstack((video, np.repeat(video[-1:], video_add_length, 0)))
        return video_complete
    f_ignore = open(less_twof_vid_path, "a+")
    video_data_bytes = lmdb_from_path.get(vid.encode())
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
    if frame_num <= clip_sec / 2:  # video duration less than clip_sec/2 seconds, abandon video
         f_ignore.write(vid+"\n")
         return np.array([])
    elif clip_sec / 2 < frame_num <= clip_sec:
        frame_add_length = int(clip_sec - frame_num)
        video_complete = np.vstack((video_gray, np.repeat(video_gray[-1:], frame_add_length, 0)))
    else:
        if frame_num >= max_clip_num:
            video_complete = video_gray[: max_clip_num]
        video_complete = normalize_video_longer_than_four_sec(video_complete, video_complete.shape[0], clip_sec)  # 4 dimensions darray

    # devide into different clips
    video_frame = video_complete.shape[0]
    video_final = []
    i = 0
    tmp_video = []
    while ((video_frame - i) // clip_sec) != 0:
        for j in range(i, i + 4):
            tmp_video.append(video_complete[j])
        tmp_video = np.concatenate(tmp_video, axis=1)
        video_final.append(tmp_video)
        tmp_video = []
        i += 4
    video_final = np.stack(video_final, axis=0)
    video_final = torch.FloatTensor(video_final)

    #  put it into the model
    video_lengths = torch.LongTensor([seq_length] * video_final.shape[0])
    if torch.cuda.is_available():
        video_final = video_final.cuda()
        video_lengths = video_lengths.cuda()
    encoder_out = model.module.get_embedding(video_final, video_lengths)
    return encoder_out.cpu().numpy()

# def save_into_lmdb(txn_video, video_feature, vid):
#     video = QQworld_Video_feature(video_feature)
#     txn_video.put(vid.encode(), pickle.dumps(video))


def main():
    parser = argparse.ArgumentParser(description="Make video lmdb database")
    parser.add_argument('--video-path', type=str, default='', help='A file containing all the video paths to be processed')
    parser.add_argument('--video-lmdb-path', type=str, default='', help='Directory to put the video lmdb of the video')
    parser.add_argument('--video_out-path', type=str, default='', help='Directory to put the video lmdb of the video')
    parser.add_argument('--lmdb-size', type=int, default=10737418240, help='Set lmdb size, default is 1GB; 64GB/68719476736')
    parser.add_argument('--model-path', type=str, default='', help='A model to get pretrained feature')
    parser.add_argument('--yaml-path', type=str, default='', help='A model initialization file')
    parser.add_argument('--less-twof-vid-path', type=str, default='', help='memorize the vid less than 2 frame')
    args = parser.parse_args()

    # Read test_config_proto.yaml
    with open(args.yaml_path, 'r', encoding='utf-8') as f_r:
        parameters = yaml.load(f_r.read(), Loader=yaml.Loader)

    # Conformer related parameters
    seq_length = parameters["seq_length"]
    input_dim = parameters["input_dim"]
    encoder_dim = parameters["encoder_dim"]
    num_encoder_layers = parameters["num_encoder_layers"]
    num_attention_heads = parameters["num_attention_heads"]
    feedforward_expansion_factor = parameters["feedforward_expansion_factor"]
    conv_expansion_factor = parameters["conv_expansion_factor"]
    input_dropout = parameters["input_dropout"]
    feedforward_dropout = parameters["feedforward_dropout"]
    attention_dropout = parameters["attention_dropout"]
    conv_dropout = parameters["conv_dropout"]
    conv_kernel_size = parameters["conv_kernel_size"]
    clip_sec = parameters["clip_sec"]
    max_frame_num = parameters["max_frame_num"]
    json_path = parameters['json_path']

    # get num_classes
    all_tag = []
    with open(json_path, "r") as json_f:
        json_dict = json.load(json_f)
    for vid, value in json_dict.items():
        if value[2] not in all_tag:
            all_tag.append(value[2])
    num_classes = len(all_tag)

    # Instantiate the model
    model = Conformer(num_classes, input_dim, encoder_dim, num_encoder_layers, num_attention_heads,
                      feedforward_expansion_factor, \
                      conv_expansion_factor, input_dropout, feedforward_dropout, attention_dropout, conv_dropout,
                      conv_kernel_size)
    model = torch.nn.DataParallel(model, device_ids=[0]).cuda()
    if not torch.cuda.is_available():
        checkpoint = torch.load(args.model_path, map_location="cpu")
    else:
        checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint)

    # get lmdb env
    video_path = args.video_path
    video_lmdb_path = args.video_lmdb_path
    video_lmdb_out_path = args.video_out_path
    video_list = get_video_lst(video_path)
    video_lmdb = lmdb.open(video_lmdb_out_path, map_size=args.lmdb_size)
    video_lmdb_from = lmdb.open(video_lmdb_path)
    txn_video_from = video_lmdb_from.begin()
    txn_video = video_lmdb.begin(write=True)

    #  get video lmdb's raw data and saved into lmdb database
    for vid in tqdm(video_list):
        video_feature = get_feature(model, vid, txn_video_from, clip_sec, seq_length, max_frame_num, args.less_twof_vid_path)
        if np.array_equal(video_feature, np.array([])):
            continue
        else:
            video = Data_Video_feature(video_feature)
            txn_video.put(vid.encode(), pickle.dumps(video))
    txn_video.commit()
    video_lmdb_from.close()
    video_lmdb.close()


if __name__ == '__main__':
    main()