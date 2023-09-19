import lmdb, argparse, torch, yaml, os, pickle, copy, torchaudio, sys
from model_Conformer import Conformer
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

class Data_Audio_feature:
    def __init__(self, audio):
        self.clip = audio.shape[0]
        self.dim = audio.shape[1]
        self.audio = audio.tobytes()

    def get_audio(self):
        audio = np.frombuffer(self.audio, dtype = np.float32)
        return audio.reshape(self.clip, self.dim)

def get_audio_lst(in_path_file):
    aids = []
    with open(in_path_file) as f_r:
        for path in f_r:
            path = path.strip()
            # basename = os.path.basename(path)
            # basename_no_extension = os.path.splitext(basename)[0]
            aids.append(path)
    return aids

def get_feature(model, aid, lmdb_from_path, clip_sec, seq_length):
    audio_data_bytes = lmdb_from_path.get(aid.encode())
    audio = pickle.loads(audio_data_bytes)
    audio_data = audio.get_audio()
    sr = int(audio.sr)
    duration = int(audio.duration)
    clip_num = duration // clip_sec

    # Compute Filter-bank features, channels set as 80, compatible with the conformer default setting
    audio_data = copy.deepcopy(audio_data)
    audio_data_torch = torch.from_numpy(audio_data).reshape(1, -1)
    fbank_feats = torch.FloatTensor([])
    for i in range(clip_num):
        feat = torchaudio.compliance.kaldi.fbank(
            waveform=audio_data_torch[:, i * sr * clip_sec: (i + 1) * sr * clip_sec],
            sample_frequency=sr,
            num_mel_bins=80,
            htk_compat=True
        )
        feat = feat.unsqueeze(0)
        fbank_feats = torch.cat((fbank_feats, feat), axis=0)

    #  put it into the model
    audio_lengths = torch.LongTensor([seq_length] * fbank_feats.shape[0])
    encoder_out = model.get_embedding(fbank_feats, audio_lengths)
    return encoder_out.cpu().numpy()


def save_into_lmdb(txn_audio, audio_feature, aid):
    audio = Data_Audio_feature(audio_feature)
    txn_audio.put(aid.encode(), pickle.dumps(audio))


def main():
    parser = argparse.ArgumentParser(description="Make audio lmdb database")
    parser.add_argument('--audio-path', type=str, default='', help='A file containing all the audio paths to be processed')
    parser.add_argument('--less-twof-aid-path', type=str, default='', help='memorize the aid less than 2 frame')
    parser.add_argument('--audio-lmdb-path', type=str, default='', help='Lmdb to get the audio data')
    parser.add_argument('--audio-out-path', type=str, default='', help='Directory to put the audio lmdb of the audio')
    parser.add_argument('--lmdb-size', type=int, default=10737418240, help='Set lmdb size, default is 1GB; 64GB/68719476736')
    parser.add_argument('--model-path', type=str, default='',  help='A model to get pretrained feature')
    parser.add_argument('--yaml-path', type=str, default='', help='A model initialization file')
    parser.add_argument('--clip-sec', type=int, default=4, help='sec per clip')
    args = parser.parse_args()

    # Read test_config_proto.yaml
    with open(args.yaml_path, 'r', encoding='utf-8') as f_r:
        parameters = yaml.load(f_r.read(), Loader=yaml.Loader)

    # Conformer related parameters
    seq_length = parameters["seq_length"]
    input_dim = parameters["input_dim"]
    num_classes = parameters["num_classes"]
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
    # Instantiate the model
    model = Conformer(num_classes, input_dim, encoder_dim, num_encoder_layers, num_attention_heads,
                      feedforward_expansion_factor, \
                      conv_expansion_factor, input_dropout, feedforward_dropout, attention_dropout, conv_dropout,
                      conv_kernel_size)

    if not torch.cuda.is_available():
        checkpoint = torch.load(args.model_path, map_location="cpu")
    else:
        checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint)

    # get lmdb env
    audio_path = args.audio_path
    lmdb_audio_out_path = args.audio_out_path
    lmdb_audio_path = args.audio_lmdb_path
    audio_list = get_audio_lst(audio_path)
    audio_lmdb_from = lmdb.open(lmdb_audio_path)
    audio_lmdb = lmdb.open(lmdb_audio_out_path, map_size = args.lmdb_size)
    txn_audio_from = audio_lmdb_from.begin()
    txn_audio = audio_lmdb.begin(write=True)

    # get aid which means the audio is too short
    f_ignore = open(args.less_twof_aid_path, "r")
    lines = f_ignore.readlines()
    f_ignore.close()
    aid_ignore = []
    for line in lines:
        line = line.strip("\n")
        # path = line.split("\t")[0]
        # aid = os.path.splitext(path.split("/")[-1])[0]
        aid_ignore.append(line)
    print(aid_ignore)
    #  get audio lmdb's raw data and saved into lmdb database
    for aid in tqdm(audio_list):
        if aid in aid_ignore:
            continue
        else:
            audio_feature = get_feature(model, aid, txn_audio_from, clip_sec, seq_length)
            save_into_lmdb(txn_audio, audio_feature, aid)



    txn_audio.commit()
    audio_lmdb_from.close()
    audio_lmdb.close()


if __name__ == '__main__':
    main()