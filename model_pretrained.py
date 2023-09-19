import numpy as np
import math
import torch.nn as nn
import torch
import torch.optim as optim
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from utils.tag_dam_loss import ClusterSoftmaxLoss
from utils.ce_match_loss import CEMatchingLoss
from utils.advloss import AdvCrossModalLoss
from torch.nn import TransformerEncoder, TransformerEncoderLayer

def cal_distance(x: torch.Tensor, y: torch.Tensor, distance_type: str = "L2", batch_size: int = 256) -> torch.Tensor:
    if distance_type == "L2":
        ones = torch.ones(batch_size, 1, requires_grad = False)
        if torch.cuda.is_available():
            ones = ones.cuda()
        x_expand = torch.kron(x, ones)
        y_expand = y.repeat(batch_size, 1)
        # dist_xy = torch.sqrt(torch.sum((x_expand - y_expand) ** 2, axis = 1))
        dist_xy = F.pairwise_distance(x_expand, y_expand)
        dist_xy = dist_xy.reshape(batch_size, batch_size)
    elif distance_type == "COS":                                    # Already L2 normalized, so this operation will obtain cosine similarity,
        dist_xy = (-1) * torch.matmul(x, y.transpose(1, 0))         # -1 indicates that larger value means larger angle between samples from x, y
    return dist_xy

class InterModalRankingLoss(nn.Module):
    def __init__(self, batch_size: int = 1024, margin: float = 0.5, distance_type: str = "L2", topk_num: int = 1024):
        """
            batch_size: int; batch size of the input tensors
            margin: float; distance margin used in triplet loss, default is 0.1
            distance_type: str; distance type ("L2"/"COS") used in triplet loss, default is "L2"
            topk_num: int; top k number for selecting most violating triplets, default is 100
        """
        super(InterModalRankingLoss, self).__init__()
        self.current_batch_size = batch_size
        self.margin = margin
        self.distance_type = distance_type
        self.topk_num = topk_num
        self.num_max_pos_pair = 1

    def forward(self, x_embed: torch.Tensor, y_embed: torch.Tensor) -> torch.Tensor:
        """
            x_embed: torch.Tensor; embedded audio features from music subnetwork, dimension is
                     [batch_num, audio_feature_dim]; audio_feature_dim: 1140
            y_embed: torch.Tensor; embedded video features from video subnetwork, dimension is
                     [batch_num, video_feature_dim]; video_feature_dim: 1024
        """

        self.current_batch_size = x_embed.shape[0]
        if self.topk_num < self.current_batch_size:
            current_topk_num = self.topk_num
        else:
            current_topk_num = self.current_batch_size


        dist_x_y_embed = cal_distance(x_embed, y_embed, distance_type = self.distance_type, batch_size = self.current_batch_size)

        aff_xy = torch.eye(self.current_batch_size, dtype = bool, requires_grad = False)
        if torch.cuda.is_available():
            aff_xy = aff_xy.cuda()

        dist_pos_pair = dist_x_y_embed.masked_fill(mask = torch.logical_not(aff_xy), value = -1 * (1e+6))
        dist_neg_pair = dist_x_y_embed.masked_fill(mask = aff_xy, value = 1 * (1e+6))

        # for top violating postive samples
        top_k_pos_pair_xy, _ = dist_pos_pair.topk(k = self.num_max_pos_pair, dim = 1)  # 求正pair中距离最大的一个（就是求对角线的值）
        top_k_pos_pair_yx, _ = dist_pos_pair.transpose(0, 1).topk(k = self.num_max_pos_pair, dim = 1) # 求正pair中距离最大的一个（就是求对角线的值）
        top_k_pos_pair_yx = top_k_pos_pair_yx.transpose(1, 0)

        # for top violating negative samples
        top_k_neg_pair_xy, _ = torch.negative(dist_neg_pair).topk(k = current_topk_num, dim = 1)
        top_k_neg_pair_xy = torch.negative(top_k_neg_pair_xy)
        dist_neg_pair_negative = torch.negative(dist_neg_pair)
        top_k_neg_pair_yx, _ = dist_neg_pair_negative.transpose(1, 0).topk(k = current_topk_num, dim = 1)
        top_k_neg_pair_yx = torch.negative(top_k_neg_pair_yx.transpose(1, 0))  # 求正pair中距离最大的topk个pair

        top_k_pos_pair_xy = torch.tile(top_k_pos_pair_xy, (1, current_topk_num))
        top_k_pos_pair_yx = torch.tile(top_k_pos_pair_yx, (current_topk_num, 1))
        shape_xy = aff_xy.shape

        top_k_pos_pair_xy = torch.reshape(top_k_pos_pair_xy, [shape_xy[0], current_topk_num, -1])
        top_k_pos_pair_yx = torch.reshape(top_k_pos_pair_yx, [-1, current_topk_num, shape_xy[1]])

        loss_xy = torch.clamp(self.margin + top_k_pos_pair_xy - top_k_neg_pair_xy.unsqueeze(2), min = 0.0)
        loss_yx = torch.clamp(self.margin + top_k_pos_pair_yx - top_k_neg_pair_yx.unsqueeze(0), min = 0.0)
        loss_xy = torch.mean(torch.reshape(loss_xy, [-1]))
        loss_yx = torch.mean(torch.reshape(loss_yx, [-1]))

        return loss_xy, loss_yx

class IntraModelStructureLoss(nn.Module):
    def __init__(self, batch_size: int = 1024, distance_type: str = "L2", k_num: int = 1024):
        """
            current_batch_size: int; batch size of the input tensors
            margin: float; distance margin used in structure loss, default is 0.1
            distance_type: str; distance type ("L2"/"COS") used in structure loss, default is "L2"
            k_num: int; number for selecting triplets, default is 100
        """
        super(IntraModelStructureLoss, self).__init__()

        self.current_batch_size = batch_size
        self.distance_type = distance_type
        self.K = k_num

    def forward(self, low_feature: torch.Tensor, embed_feature: torch.Tensor, id_numpy: torch.Tensor) -> torch.Tensor:
        """
            low_feature: torch.Tensor; low audio/video features from music/video subnetwork, dimension is
                         [batch_num, feature_dim]
            embed_feature: torch.Tensor; embedded audio/video features from music/video subnetwork, dimension is
                           [batch_num, video_feature_dim]
            id_numpy: torch.Tensor; id list
        """
        id_numpy = id_numpy.reshape(id_numpy.shape[0], ).tolist()
        id_numpy_no_rep = list(set(id_numpy))
        index = [id_numpy.index(id) for id in id_numpy_no_rep]
        low_feature = low_feature[index]
        embed_feature = embed_feature[index]
        self.current_batch_size = low_feature.shape[0]
        if self.K < self.current_batch_size:
            current_K = self.K
        else:
            current_K = self.current_batch_size
        low_feature_l2_normalize = F.normalize(low_feature, p = 2)
        embed_feature_l2_normalize = F.normalize(embed_feature, p = 2)

        dist_low = cal_distance(low_feature_l2_normalize, low_feature_l2_normalize, distance_type = self.distance_type,
                                batch_size = self.current_batch_size)
        dist_embed = cal_distance(embed_feature_l2_normalize, embed_feature_l2_normalize,
                                  distance_type = self.distance_type, batch_size = self.current_batch_size)

        # Select two slices of size (current_K, current_K - 1) to calculate the distance difference
        x_start = np.random.choice(range(self.current_batch_size - current_K + 1), size = 1)[0]
        y_start = np.random.choice(range(self.current_batch_size - current_K + 1), size = 1)[0]
        dist_low_select1 = dist_low[x_start : (x_start + current_K - 1), y_start : (y_start + current_K - 2)]
        dist_low_select2 = dist_low[x_start : (x_start + current_K - 1), (y_start + 1) : (y_start + current_K - 1)]
        dist_embed_select1 = dist_embed[x_start : (x_start + current_K - 1), y_start : (y_start + current_K - 2)]
        dist_embed_select2 = dist_embed[x_start : (x_start + current_K - 1), (y_start + 1) : (y_start + current_K - 1)]

        # Calculate intra-modal structure loss (xi, xj, xk or yi, yj, yk)
        coeff = torch.sign(dist_low_select1 - dist_low_select2) - torch.sign(dist_embed_select1 - dist_embed_select2)
        dist_embed_inverse = dist_embed_select2 - dist_embed_select1
        dist_tensor = torch.mul(coeff, dist_embed_inverse)
        #  loss_structure = torch.sum(torch.mean(dist_tensor, axis = 1))
        loss_structure = torch.mean(torch.reshape(dist_tensor, [-1]))
        return loss_structure



class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0., init_method = "kaiming"):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        # Initialize linear layer weights
        self.init_linear_weights(self.net, init_method)

    def init_linear_weights(self, model, init_method):
        for m in model:
            if isinstance(m, nn.Linear):
                if init_method == "xavier":
                    nn.init.xavier_normal_(m.weight)
                elif init_method == "kaiming":
                    nn.init.kaiming_normal_(m.weight)
                m.bias.data.fill_(0.01)

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, init_method = "kaiming", dropout = 0.):
        super().__init__()
        inner_dim = dim_head * heads # 3
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False) # 1->9
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
        # Initialize linear layer weights
        self.init_linear_weights(nn.Sequential(self.to_qkv), init_method, bias = False)
        self.init_linear_weights(self.to_out, init_method)

    def init_linear_weights(self, model, init_method, bias = True):
        for m in model:
            if isinstance(m, nn.Linear):
                if init_method == "xavier":
                    nn.init.xavier_normal_(m.weight)
                elif init_method == "kaiming":
                    nn.init.kaiming_normal_(m.weight)
                if bias == True:
                    m.bias.data.fill_(0.01)

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int = 512, max_len: int = 7) -> None:
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model, requires_grad = False)
        position = torch.arange(0, max_len, dtype = torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, length: int):
        return self.pe[:, :length]



class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dim_out, dropout = 0., init_method = "kaiming"):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, init_method = init_method, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout, init_method = init_method))
            ]))
        self.final_linear = nn.Linear(dim, dim_out)

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        x = self.final_linear(x)
        return x

class EmbeddingNet(nn.Module):
    def __init__(self, input_size, output_size, dropout, use_bn, momentum = 0.99, hidden_size = 0, init_method = "kaiming"):
        super(EmbeddingNet, self).__init__()
        self.init_method = init_method
        modules = []
        if hidden_size > 0:
            modules.append(nn.Linear(in_features = input_size, out_features = hidden_size))
            if use_bn:
                modules.append(nn.BatchNorm1d(num_features = hidden_size))
            modules.append(nn.Sigmoid())
            modules.append(nn.Linear(in_features = hidden_size, out_features = output_size))
            modules.append(nn.BatchNorm1d(num_features = output_size, momentum = momentum))
        else:
            modules.append(nn.Linear(in_features = input_size, out_features = output_size))
            modules.append(nn.BatchNorm1d(num_features = output_size))
        modules.append(nn.ReLU())
        modules.append(nn.Linear(output_size, output_size))
        self.net = nn.Sequential(*modules)
        # Initialize linear layer weights
        self.init_linear_weights(self.net, self.init_method)

    def init_linear_weights(self, model, init_method):
        for m in model:
            if isinstance(m, nn.Linear):
                if init_method == "xavier":
                    nn.init.xavier_normal_(m.weight)
                elif init_method == "kaiming":
                    nn.init.kaiming_normal_(m.weight)
                m.bias.data.fill_(0.01)

    def forward(self, x):
        output = self.net(x)
        return output

    def get_embedding(self, x):
        return self.forward(x)

class RhythmEmbedding(nn.Module):
    def __init__(self, embed_list=[18, 49, 65]):         # embed_list default is [17, 129, 65]
        super(RhythmEmbedding, self).__init__()
        self.n_beat_class = embed_list[0]
        self.beat_strength = embed_list[1]
        self.beat_interval_width = embed_list[2]
        self.n_beat_embed = nn.Embedding(self.n_beat_class, 256)
        self.beat_strength_embed = nn.Embedding(self.beat_strength, 128)
        self.beat_interval_width_embed = nn.Embedding(self.beat_interval_width, 128)

    def forward(self, x):
        n_beats_seq = x[:, :, 0]
        beat_strength_seq = x[:, :, 1]
        beat_interval_width_seq = x[:, :, 2]
        n_beats_seq_embedding = self.n_beat_embed(n_beats_seq)
        beat_strength_seq_embedding = self.beat_strength_embed(beat_strength_seq)
        beat_interval_width_seq_embedding = self.beat_interval_width_embed(beat_interval_width_seq)
        rhythm_seq_embedding = torch.concat([n_beats_seq_embedding, beat_strength_seq_embedding,
                                             beat_interval_width_seq_embedding], axis = 2)
        # rhythm_seq_embedding = rhythm_seq_embedding.reshape(x.shape[0], -1)
        return rhythm_seq_embedding

class OpticalFlowEmbedding(nn.Module):
    def __init__(self, embed_class=65):        # embed_class defalut is 65
        super(OpticalFlowEmbedding, self).__init__()
        self.optic_displacement = embed_class
        self.optic_displacement_embed = nn.Embedding(self.optic_displacement, 512)

    def forward(self, x):
        optic_displacement_seq_embedding = self.optic_displacement_embed(x)
        return optic_displacement_seq_embedding

class Model_structure(nn.Module):
    def __init__(self, a_data_dim, v_data_dim, t_data_dim, dim_theta, dim_rho, init_method, \
        encoder_hidden_size, decoder_hidden_size, depth_transformer, additional_dropout, \
        momentum, first_additional_triplet, second_additional_triplet, dropout_encoder, \
        dropout_decoder, margin, lr, target_dim, keep_prob, loss_mask, max_grad_norm, decay_rate, \
        inter_margin, inter_topk, intra_topk, batch_size, distance_type, a_enc_dmodel, v_enc_dmodel, \
        nlayers, dropout_enc, enc_hidden_dim, cross_hidden_dim, max_clip, add_rhythm):
        super(Model_structure, self).__init__()

        print('Initializing model variables...', end = '')
        # model structure
        # Dimension of embedding
        self.v_data_dim = v_data_dim
        self.a_data_dim = a_data_dim
        self.t_data_dim = t_data_dim
        self.r_enc = dropout_encoder
        self.r_proj = dropout_decoder
        self.dim_theta = dim_theta
        self.dim_rho = dim_rho
        self.init_method = init_method
        self.hidden_size_encoder = encoder_hidden_size
        self.hidden_size_decoder = decoder_hidden_size
        self.depth_transformer = depth_transformer
        self.r_dec = additional_dropout
        self.momentum = momentum
        self.first_additional_triplet = first_additional_triplet
        self.second_additional_triplet = second_additional_triplet
        self.margin = margin
        self.lr = lr
        self.target_dim = target_dim
        self.keep_prob = keep_prob
        self.loss_mask = loss_mask
        self.max_grad_norm = max_grad_norm
        self.decay_rate = decay_rate
        self.a_enc_dmodel = a_enc_dmodel
        self.v_enc_dmodel = v_enc_dmodel
        self.max_clip = max_clip
        self.nlayers = nlayers
        self.dropout_enc = dropout_enc
        self.enc_hidden_dim = enc_hidden_dim
        self.cross_hidden_dim = cross_hidden_dim
        self.add_rhythm = add_rhythm

        print('Initializing trainable models...', end = '')
        if self.add_rhythm == 1:
            self.A_rhythm_embed = RhythmEmbedding()
            self.V_optic_embed = OpticalFlowEmbedding()
        self.A_p_embed = PositionalEncoding(max_len=self.max_clip, d_model=self.a_enc_dmodel)
        self.V_p_embed = PositionalEncoding(max_len=self.max_clip, d_model=self.v_enc_dmodel)
        self.A_enc_former = Transformer(self.a_enc_dmodel, self.nlayers, 3, 1, self.enc_hidden_dim, self.a_data_dim,
                                        self.dropout_enc, self.init_method)
        self.V_enc_former = Transformer(self.v_enc_dmodel, self.nlayers, 3, 1, self.enc_hidden_dim, self.v_data_dim,
                                        self.dropout_enc, self.init_method)

        self.A_enc = EmbeddingNet(
            input_size = self.a_data_dim,
            hidden_size = self.hidden_size_encoder,
            output_size = self.dim_rho,
            dropout = self.r_enc,
            use_bn = True,
            momentum = self.momentum,
            init_method = self.init_method
        )
        self.V_enc = EmbeddingNet(
            input_size = v_data_dim,
            hidden_size = self.hidden_size_encoder,
            output_size = self.dim_rho,
            dropout = self.r_enc,
            use_bn = True,
            momentum = self.momentum,
            init_method = self.init_method
        )
        self.cross_attention = Transformer(self.dim_rho, self.depth_transformer, 3, 100, self.cross_hidden_dim,
                                           self.dim_rho, self.r_enc, self.init_method)

        self.W_proj = EmbeddingNet(
            input_size = t_data_dim,
            output_size = self.dim_theta,
            dropout = self.r_dec,
            use_bn = True,
            momentum = self.momentum,
            init_method = self.init_method
        )

        self.D = EmbeddingNet(
            input_size = self.dim_theta,
            output_size = self.dim_rho,
            dropout = self.r_dec,
            use_bn = True,
            momentum = self.momentum,
            init_method = self.init_method
        )

        self.A_proj = EmbeddingNet(input_size = self.dim_rho, hidden_size = self.hidden_size_decoder,
                                   output_size = self.dim_theta,
            dropout = self.r_proj, momentum = self.momentum, use_bn = True, init_method = self.init_method)

        self.V_proj = EmbeddingNet(input_size = self.dim_rho, hidden_size = self.hidden_size_decoder,
                                   output_size = self.dim_theta,
            dropout = self.r_proj, momentum = self.momentum, use_bn = True, init_method = self.init_method)

        self.A_rec = EmbeddingNet(input_size = self.dim_theta, output_size = self.dim_rho, dropout = self.r_dec,
                                  momentum = self.momentum,
            use_bn = True, init_method = self.init_method)

        self.V_rec = EmbeddingNet(input_size = self.dim_theta, output_size = self.dim_rho, dropout = self.r_dec,
                                  momentum = self.momentum,
            use_bn = True, init_method = self.init_method)

        self.pos_emb1D = torch.nn.Parameter(torch.randn(2, self.dim_rho))

        # Loss function
        print('Defining losses...', end = '')
        self.criterion_reg = nn.MSELoss()
        self.triplet_loss = nn.TripletMarginLoss(margin = self.margin)
        print('Done')

        # addtional loss moduels
        self.damsoftmax_loss = ClusterSoftmaxLoss(input_dim=self.dim_rho, target_dim=self.target_dim, \
                                                  drop_rate=1 - self.keep_prob, init_method=self.init_method)
        self.ce_match_loss = CEMatchingLoss(self.dim_rho, 1 - self.keep_prob, self.init_method)
        self.adv_loss = AdvCrossModalLoss(self.dim_rho)
        self.interloss = InterModalRankingLoss(batch_size, inter_margin, distance_type, inter_topk)
        self.intraloss = IntraModelStructureLoss(batch_size, distance_type, intra_topk)

        # optimizer and scheduler
        self.params = (list(self.A_enc_former.parameters()) + list(self.V_enc_former.parameters()) +
                       list(self.A_proj.parameters()) + list(self.V_proj.parameters()) +
                       list(self.A_rec.parameters()) + list(self.V_rec.parameters()) +
                       list(self.V_enc.parameters()) + list(self.A_enc.parameters()) +
                       list(self.cross_attention.parameters()) + list(self.D.parameters()) +
                       list(self.W_proj.parameters()))
        if self.loss_mask[4] == 1:
            self.params += list(self.damsoftmax_loss.parameters())
        if self.loss_mask[5] == 1:
            self.params += list(self.ce_match_loss.parameters())
        if self.loss_mask[6] == 1:
            self.params += list(self.adv_loss.parameters())
        if self.add_rhythm == 1:
            self.params += list(self.A_rhythm_embed.parameters()) + list(self.V_optic_embed.parameters())

        self.optimizer = optim.Adam(self.params, lr=float(self.lr))
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=decay_rate)

    def optimize_scheduler(self):
        self.scheduler.step()

    def forward(self, audio, image, word_embedding, n_audio, n_image, n_word_embedding, p_label, p_aid_numpy, p_vid_numpy,
                p_rhythm, p_optical, n_rhythm, n_optical, if_testing):
        if self.add_rhythm == 1:
            audio += self.A_rhythm_embed(p_rhythm)
            image += self.V_optic_embed(p_optical)
            n_audio += self.A_rhythm_embed(n_rhythm)
            n_image += self.V_optic_embed(n_optical)
        audio += self.A_p_embed(self.max_clip).repeat(audio.shape[0], 1, 1)
        image += self.V_p_embed(self.max_clip).repeat(image.shape[0], 1, 1)
        audio = self.A_enc_former(audio)
        image = self.V_enc_former(image)
        audio = audio.mean(axis=1)
        image = image.mean(axis=1)
        audio = torch.squeeze(audio, axis=1)
        image = torch.squeeze(image, axis=1)
        n_audio += self.A_p_embed(self.max_clip).repeat(n_audio.shape[0], 1, 1)
        n_image += self.V_p_embed(self.max_clip).repeat(n_image.shape[0], 1, 1)
        n_audio = self.A_enc_former(n_audio)
        n_image = self.V_enc_former(n_image)
        n_audio = n_audio.mean(axis=1)
        n_image = n_image.mean(axis=1)
        n_audio = torch.squeeze(n_audio, axis=1)
        n_image = torch.squeeze(n_image, axis=1)
        self.if_testing = if_testing
        self.p_label = p_label
        self.p_aid_numpy = p_aid_numpy
        self.p_vid_numpy = p_vid_numpy
        self.phi_a = self.A_enc(audio)
        self.phi_v = self.V_enc(image)
        self.phi_a_neg = self.A_enc(n_audio)
        self.phi_v_neg = self.V_enc(n_image)
        self.w = word_embedding
        self.w_neg = n_word_embedding
        self.theta_w = self.W_proj(word_embedding)
        self.theta_w_neg = self.W_proj(n_word_embedding)
        self.rho_w = self.D(self.theta_w)
        self.rho_w_neg = self.D(self.theta_w_neg)
        self.positive_input = torch.stack((self.phi_a + self.pos_emb1D[0, :], self.phi_v + self.pos_emb1D[1, :]), dim=1)
        self.negative_input = torch.stack(
            (self.phi_a_neg + self.pos_emb1D[0, :], self.phi_v_neg + self.pos_emb1D[1, :]), dim=1)
        self.phi_attn = self.cross_attention(self.positive_input)
        self.phi_attn_neg = self.cross_attention(self.negative_input)
        self.audio_fe_attn = self.phi_a + self.phi_attn[:, 0, :]
        self.video_fe_attn = self.phi_v + self.phi_attn[:, 1, :]
        self.audio_fe_neg_attn = self.phi_a_neg + self.phi_attn_neg[:, 0, :]
        self.video_fe_neg_attn = self.phi_v_neg + self.phi_attn_neg[:, 1, :]
        self.theta_v = self.V_proj(self.video_fe_attn)
        self.theta_v_neg = self.V_proj(self.video_fe_neg_attn)
        self.theta_a = self.A_proj(self.audio_fe_attn)
        self.theta_a_neg = self.A_proj(self.audio_fe_neg_attn)
        self.phi_v_rec = self.V_rec(self.theta_v)
        self.phi_a_rec = self.A_rec(self.theta_a)
        self.rho_a = self.D(self.theta_a)
        self.rho_a_neg = self.D(self.theta_a_neg)
        self.rho_v = self.D(self.theta_v)
        self.rho_v_neg = self.D(self.theta_v_neg)
        self.theta_v = F.normalize(self.theta_v, p=2, dim=1)
        self.theta_a = F.normalize(self.theta_a, p=2, dim=1)
        self.rho_v = F.normalize(self.rho_v, p=2, dim=1)
        self.rho_a = F.normalize(self.rho_a, p=2, dim=1)
        self.theta_w = F.normalize(self.theta_w, p=2, dim=1)
        self.rho_w = F.normalize(self.rho_w, p=2, dim=1)

    def backward(self, optimize):

        loss_gen = 0

        if self.loss_mask[0] >= 0:
            first_pair = self.first_additional_triplet * (
                    self.triplet_loss(self.theta_a, self.theta_w, self.theta_a_neg) + \
                    self.triplet_loss(self.theta_v, self.theta_w, self.theta_v_neg)
            )
            second_pair = self.second_additional_triplet * (
                    self.triplet_loss(self.theta_w, self.theta_a, self.theta_a_neg) + \
                    self.triplet_loss(self.theta_w, self.theta_v, self.theta_v_neg)
            )
            # loss_theta_awa, loss_theta_waw = self.interloss(self.theta_a, self.theta_w, self.p_aid_numpy, self.p_label, self.aid2tag, "aw")
            # loss_theta_wvw, loss_theta_vwv = self.interloss(self.theta_w, self.theta_v, self.p_label, self.p_vid_numpy, self.tag2vid, "wv")
            # first_pair = self.first_additional_triplet * (loss_theta_awa + loss_theta_vwv)
            # second_pair = self.second_additional_triplet * (loss_theta_waw + loss_theta_wvw)
            l_t = first_pair + second_pair
            loss_gen += l_t * self.loss_mask[0]

        if self.loss_mask[1] >= 0:
            l_r = self.criterion_reg(self.phi_v_rec, self.phi_v) + \
                  self.criterion_reg(self.phi_a_rec, self.phi_a) + \
                  self.criterion_reg(self.theta_v, self.theta_w) + \
                  self.criterion_reg(self.theta_a, self.theta_w)
            loss_gen += l_r * self.loss_mask[1]

        if self.loss_mask[2] >= 0:
            l_rec = self.criterion_reg(self.w, self.rho_v) + \
                    self.criterion_reg(self.w, self.rho_a) + \
                    self.criterion_reg(self.w, self.rho_w)

            # l_cwv, _ = self.interloss(self.rho_w, self.rho_v)
            # _, l_cwa = self.interloss(self.rho_a, self.rho_w)
            # l_cwv, _ = self.interloss(self.theta_w, self.rho_v)
            # _, l_cwa = self.interloss(self.rho_a, self.rho_w)
            # l_cwa = self.interloss(self.rho_w, self.rho_a)
            # l_cwv = self.interloss(self.rho_w, self.rho_v)
            # l_cw = l_cwa + l_cwv
            # l_cmd = l_rec + l_cw
            l_cmd = l_rec
            loss_gen += l_cmd * self.loss_mask[2]

        # if self.loss_mask[3] >= 0:
        #     l_wv, l_vw = self.interloss(self.theta_w, self.theta_v)
        #     l_aw, l_wa = self.interloss(self.theta_a, self.theta_w)
        #     l_w = l_wa + l_aw + l_wv + l_vw
        #     loss_gen += l_w * self.loss_mask[3]

        # if self.loss_mask[4] >= 0:
        #     damloss, dam_acc = self.damsoftmax_loss(self.rho_a, self.rho_v, self.p_label, self.if_testing)
        #     loss_gen += damloss * self.loss_mask[4]
        #

        if self.loss_mask[3] >= 0:
            ce_match, ce_acc = self.ce_match_loss(self.rho_a, self.rho_v, self.rho_a_neg, self.rho_v_neg)
            loss_gen += ce_match * self.loss_mask[5]

        # if self.loss_mask[6] >= 0:
        #     advloss, adv_acc = self.adv_loss(self.rho_a, self.rho_v)
        #     loss_gen -= advloss * self.loss_mask[6]

        if self.loss_mask[4] >= 0:
            loss_theta_av, loss_theta_va = self.interloss(self.theta_a, self.theta_v)
            # loss_rho_av, loss_rho_va = self.interloss(self.rho_a, self.rho_v)
            # loss_av = loss_rho_av + loss_theta_av + loss_rho_va + loss_theta_va
            loss_av = loss_theta_av + loss_theta_va
            loss_gen += loss_av * self.loss_mask[7]

        # intra loss
        # if self.loss_mask[8] >= 0:
        #     loss_structure_a_phi = self.intraloss(self.phi_a, self.audio_fe_attn, self.p_aid_numpy)
        #     loss_structure_a_theta = self.intraloss(self.audio_fe_attn, self.phi_a_rec, self.p_aid_numpy)
        #     loss_structure_v_phi = self.intraloss(self.phi_v, self.video_fe_attn, self.p_vid_numpy)
        #     loss_structure_v_theta = self.intraloss(self.video_fe_attn, self.phi_v_rec, self.p_vid_numpy)
        #     l_stru = loss_structure_a_phi + loss_structure_a_theta + loss_structure_v_phi + loss_structure_v_theta
        #     loss_gen += l_stru * self.loss_mask[8]

        if optimize == True:
            self.optimizer.zero_grad()
            loss_gen.backward()
            nn.utils.clip_grad_norm_(self.params, self.max_grad_norm)
            self.optimizer.step()

        loss = {'aut_enc': 0, 'gen_cyc': 0,
                'gen_reg': 0, 'gen': loss_gen}

        loss_numeric = loss['gen_cyc'] + loss['gen']

        return loss_numeric, loss, ce_acc

    def optimize_params(self, audio, video, cls_embedding, n_audio, n_video, n_cls_embedding, p_label, p_aid_numpy,
                        p_vid_numpy, p_rhythm, p_optical, n_rhythm, n_optical, scheduler_state=None, optimizer_state=None,
                        optimize=False, if_testing=False):

        # audio_negative = F.normalize(audio_negative, p=2, dim=1)
        # video_negative = F.normalize(video_negative, p=2, dim=1)
        # cls_embedding = F.normalize(cls_embedding, p=2, dim=1)
        # negative_cls_embedding = F.normalize(negative_cls_embedding, p=2, dim=1)
        if str(type(scheduler_state)) == "<class 'numpy.ndarray'>":
            self.scheduler.load_state_dict(scheduler_state)
        if str(type(optimizer_state)) == "<class 'numpy.ndarray'>":
            self.optimizer.load_state_dict(optimizer_state)

        self.forward(audio, video, cls_embedding, n_audio, n_video, n_cls_embedding, p_label, p_aid_numpy, p_vid_numpy,
                     p_rhythm, p_optical, n_rhythm, n_optical, \
                     if_testing)

        loss_numeric, loss, ce_acc = self.backward(optimize)

        return loss_numeric, loss, ce_acc
    def get_embeddings(self, audio = None, video = None, rhythm = None, optical = None, text = None, q=1):
        if text != None:
            text = F.normalize(text, p=2, dim=1)
            theta_w = self.W_proj(text)
            rho_w = self.D(theta_w)
            theta_w = F.normalize(theta_w, p=2, dim=1)
            rho_w = F.normalize(rho_w, p=2, dim=1)
            return rho_w
        else:
            audio = F.normalize(audio, p=2, dim=1)
            video = F.normalize(video, p=2, dim=1)
            if self.add_rhythm == 1:
                audio += self.A_rhythm_embed(rhythm)
                video += self.V_optic_embed(optical)
            audio += self.A_p_embed(self.max_clip).repeat(audio.shape[0], 1, 1)
            video += self.V_p_embed(self.max_clip).repeat(video.shape[0], 1, 1)
            audio = self.A_enc_former(audio)
            video = self.V_enc_former(video)
            audio = audio.mean(axis=1)
            video = video.mean(axis=1)
            audio = torch.squeeze(audio, axis=1)
            video = torch.squeeze(video, axis=1)
            phi_a = self.A_enc(audio)
            phi_v = self.V_enc(video)
            positive_input = torch.stack(
                (phi_a + self.pos_emb1D[0, :], phi_v.repeat(phi_a.shape[0], 1) + self.pos_emb1D[1, :]),
                dim=1)
            phi_attn = self.cross_attention(positive_input)
            audio_fe_attn = phi_a + phi_attn[:, 0, :]
            video_fe_attn = torch.mean(phi_v.repeat(phi_a.shape[0], 1) + phi_attn[:, 1, :], keepdim=True, dim=0)
            theta_a = self.A_proj(audio_fe_attn)
            theta_v = self.V_proj(video_fe_attn)
            print(theta_v)
            rho_v = self.D(theta_v)
            rho_a = self.D(theta_a)
            theta_v = F.normalize(theta_v, p=2, dim=1)
            theta_a = F.normalize(theta_a, p=2, dim=1)
            rho_v = F.normalize(rho_v, p=2, dim=1)
            rho_a = F.normalize(rho_a, p=2, dim=1)

            return theta_a, theta_v, rho_v, rho_a
