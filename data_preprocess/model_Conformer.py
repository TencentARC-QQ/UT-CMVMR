# flake8: noqa
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch.optim as optim
from typing import Optional
from typing import Tuple

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x * x.sigmoid()

class GLU(nn.Module):
    def __init__(self, dim: int) -> None:
        super(GLU, self).__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        outputs, gate = x.chunk(2, dim = self.dim)
        return outputs * gate.sigmoid()

class FeedForwardModule(nn.Module):
    def __init__(self, encoder_dim: int = 512, expansion_factor: int = 4, dropout: float = 0.1) -> None:
        super(FeedForwardModule, self).__init__()
        self.sequential = nn.Sequential(
            nn.LayerNorm(encoder_dim),
            nn.Linear(encoder_dim, encoder_dim * expansion_factor),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(encoder_dim * expansion_factor, encoder_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.sequential(x)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int = 512, max_len: int = 10000) -> None:
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model, requires_grad = False)
        position = torch.arange(0, max_len, dtype = torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, length: int) -> Tensor:
        return self.pe[:, :length]

class RelativeMultiHeadAttentionModule(nn.Module):
    def __init__(self, d_model: int = 512, num_heads: int = 8, dropout: float = 0.1):
        super(RelativeMultiHeadAttentionModule, self).__init__()
        assert d_model % num_heads == 0, "d_model % num_heads should be zero."
        self.d_model = d_model
        self.d_head = int(d_model / num_heads)
        self.num_heads = num_heads
        self.sqrt_dim = math.sqrt(d_model)

        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.pos_proj = nn.Linear(d_model, d_model, bias = False)

        self.dropout = nn.Dropout(p = dropout)

        self.u_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        self.v_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        torch.nn.init.xavier_uniform_(self.u_bias)
        torch.nn.init.xavier_uniform_(self.v_bias)

        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, pos_embedding: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        batch_size = value.size(0)

        query = self.query_proj(query).view(batch_size, -1, self.num_heads, self.d_head)
        key = self.key_proj(key).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        value = self.value_proj(value).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        pos_embedding = self.pos_proj(pos_embedding).view(batch_size, -1, self.num_heads, self.d_head)

        content_score = torch.matmul((query + self.u_bias).transpose(1, 2), key.transpose(2, 3))
        pos_score = torch.matmul((query + self.v_bias).transpose(1, 2), pos_embedding.permute(0, 2, 3, 1))
        pos_score = self._relative_shift(pos_score)

        score = (content_score + pos_score) / self.sqrt_dim

        if mask is not None:
            mask = mask.unsqueeze(1)
            score.masked_fill_(mask, -1e9)

        attn = F.softmax(score, -1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, value).transpose(1, 2)
        context = context.contiguous().view(batch_size, -1, self.d_model)

        return self.out_proj(context)

    def _relative_shift(self, pos_score: Tensor) -> Tensor:
        batch_size, num_heads, seq_length1, seq_length2 = pos_score.size()
        zeros = pos_score.new_zeros(batch_size, num_heads, seq_length1, 1)
        padded_pos_score = torch.cat([zeros, pos_score], dim = -1)

        padded_pos_score = padded_pos_score.view(batch_size, num_heads, seq_length2 + 1, seq_length1)
        pos_score = padded_pos_score[:, :, 1:].view_as(pos_score)

        return pos_score

class MultiHeadedSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super(MultiHeadedSelfAttention, self).__init__()
        self.positional_encoding = PositionalEncoding(d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.attention = RelativeMultiHeadAttentionModule(d_model, num_heads, dropout)
        self.dropout = nn.Dropout(p = dropout)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None):
        batch_size, seq_length, _ = x.size()
        pos_embedding = self.positional_encoding(seq_length)
        pos_embedding = pos_embedding.repeat(batch_size, 1, 1)

        x = self.layer_norm(x)
        outputs = self.attention(x, x, x, pos_embedding = pos_embedding, mask = mask)

        return self.dropout(outputs)

class Transpose(nn.Module):
    def __init__(self, shape: tuple):
        super(Transpose, self).__init__()
        self.shape = shape

    def forward(self, x: Tensor) -> Tensor:
        return x.transpose(*self.shape)

class DepthwiseConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, \
        bias: bool = False) -> None:
        super(DepthwiseConv1d, self).__init__()
        assert out_channels % in_channels == 0, "out_channels should be constant multiple of in_channels"
        self.conv = nn.Conv1d(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = kernel_size,
            groups = in_channels,
            stride = stride,
            padding = padding,
            bias = bias
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)

class PointwiseConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, padding: int = 0, bias: bool = True) -> None:
        super(PointwiseConv1d, self).__init__()
        self.conv = nn.Conv1d(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = 1,
            stride = stride,
            padding = padding,
            bias = bias
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)

class ConformerConvModule(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int = 31, expansion_factor: int = 2, dropout: float = 0.1) -> None:
        super(ConformerConvModule, self).__init__()
        assert (kernel_size - 1) % 2 == 0, "kernel_size should be an odd number for 'SAME' padding"
        assert expansion_factor == 2, "Currently, Only Supports expansion_factor 2"

        self.sequential = nn.Sequential(
            nn.LayerNorm(in_channels),
            Transpose(shape = (1, 2)),
            PointwiseConv1d(in_channels, in_channels * expansion_factor, stride = 1, padding = 0, bias = True),
            GLU(dim = 1),
            DepthwiseConv1d(in_channels, in_channels, kernel_size, stride = 1, padding = (kernel_size - 1) // 2),
            nn.BatchNorm1d(in_channels),
            Swish(),
            PointwiseConv1d(in_channels, in_channels, stride = 1, padding = 0, bias = True),
            nn.Dropout(p = dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.sequential(x).transpose(1, 2)

class Conv2dSubampling(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(Conv2dSubampling, self).__init__()
        self.sequential = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = 2),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 2),
            nn.ReLU(),
        )

    def forward(self, x: Tensor, input_lengths: Tensor) -> Tuple[Tensor, Tensor]:
        outputs = self.sequential(x.unsqueeze(1))
        batch_size, channels, subsampled_lengths, sumsampled_dim = outputs.size()

        outputs = outputs.permute(0, 2, 1, 3)
        outputs = outputs.contiguous().view(batch_size, subsampled_lengths, channels * sumsampled_dim)

        output_lengths = input_lengths >> 2
        output_lengths -= 1

        return outputs, output_lengths

class ResidualConnectionModule(nn.Module):
    def __init__(self, module: nn.Module, module_factor: float = 1.0, input_factor: float = 1.0):
        super(ResidualConnectionModule, self).__init__()
        self.module = module
        self.module_factor = module_factor
        self.input_factor = input_factor

    def forward(self, x: Tensor) -> Tensor:
        return (self.module(x) * self.module_factor) + (x * self.input_factor)

class ConformerBlock(nn.Module):
    def __init__(self, encoder_dim: int = 512, num_attention_heads: int = 8, feedforward_expansion_factor: int = 4, \
        conv_expansion_factor: int = 2, feedforward_dropout: float = 0.1, attention_dropout: float = 0.1, \
        conv_dropout: float = 0.1, conv_kernel_size: int = 31):
        super(ConformerBlock, self).__init__()

        self.feedforward_residual_factor = 0.5

        self.sequential = nn.Sequential(
            ResidualConnectionModule(
                module = FeedForwardModule(
                    encoder_dim = encoder_dim,
                    expansion_factor = feedforward_expansion_factor,
                    dropout = feedforward_dropout,
                ),
                module_factor = self.feedforward_residual_factor
            ),
            ResidualConnectionModule(
                module = MultiHeadedSelfAttention(
                    d_model = encoder_dim,
                    num_heads = num_attention_heads,
                    dropout = attention_dropout,
                )
            ),
            ResidualConnectionModule(
                module = ConformerConvModule(
                    in_channels = encoder_dim,
                    kernel_size = conv_kernel_size,
                    expansion_factor = conv_expansion_factor,
                    dropout = conv_dropout,
                )
            ),
            ResidualConnectionModule(
                module = FeedForwardModule(
                    encoder_dim = encoder_dim,
                    expansion_factor = feedforward_expansion_factor,
                    dropout = feedforward_dropout,
                ),
                module_factor = self.feedforward_residual_factor
            ),
            nn.LayerNorm(encoder_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.sequential(x)

class ConformerEncoder(nn.Module):
    def __init__(self, input_dim: int = 80, encoder_dim: int = 512, num_layers: int = 6, num_attention_heads: int = 8, \
        feedforward_expansion_factor: int = 4, conv_expansion_factor: int = 2, input_dropout: float = 0.1, \
        feedforward_dropout: float = 0.1, attention_dropout: float = 0.1, conv_dropout: float = 0.1, conv_kernel_size: int = 31):
        super(ConformerEncoder, self).__init__()
        self.conv_subsample = Conv2dSubampling(in_channels = 1, out_channels = encoder_dim)
        self.input_projection = nn.Sequential(
            nn.Linear(encoder_dim * (((input_dim - 1) // 2 - 1) // 2), encoder_dim),
            nn.Dropout(p = input_dropout),
        )
        self.layers = nn.ModuleList([ConformerBlock(
            encoder_dim = encoder_dim,
            num_attention_heads = num_attention_heads,
            feedforward_expansion_factor = feedforward_expansion_factor,
            conv_expansion_factor = conv_expansion_factor,
            feedforward_dropout = feedforward_dropout,
            attention_dropout = attention_dropout,
            conv_dropout = conv_dropout,
            conv_kernel_size = conv_kernel_size
        ) for _ in range(num_layers)])

    def count_parameters(self) -> int:
        return sum([p.numel() for p in self.parameters()])

    def update_dropout(self, dropout: float) -> None:
        for name, child in self.named_children():
            if isinstance(child, nn.Dropout):
                child.p = dropout

    def forward(self, x: Tensor, input_lengths: Tensor) -> Tuple[Tensor, Tensor]:
        outputs, output_lengths = self.conv_subsample(x, input_lengths)
        outputs = self.input_projection(outputs)

        for layer in self.layers:
            outputs = layer(outputs)

        return outputs, output_lengths

class Conformer(nn.Module):
    def __init__(self, num_classes: int, input_dim: int = 80, encoder_dim: int = 512, num_encoder_layers: int = 6, \
        num_attention_heads: int = 8, feedforward_expansion_factor: int = 4, conv_expansion_factor: int = 2, \
        input_dropout: float = 0.1, feedforward_dropout: float = 0.1, attention_dropout: float = 0.1, \
        conv_dropout: float = 0.1, conv_kernel_size: int = 31) -> None:
        super(Conformer, self).__init__()
        self.encoder = ConformerEncoder(
            input_dim = input_dim,
            encoder_dim = encoder_dim,
            num_layers = num_encoder_layers,
            num_attention_heads = num_attention_heads,
            feedforward_expansion_factor = feedforward_expansion_factor,
            conv_expansion_factor = conv_expansion_factor,
            input_dropout = input_dropout,
            feedforward_dropout = feedforward_dropout,
            attention_dropout = attention_dropout,
            conv_dropout = conv_dropout,
            conv_kernel_size = conv_kernel_size,
        )
        self.classification = nn.Linear(encoder_dim, num_classes, bias = False)

    def count_parameters(self) -> int:
        return self.encoder.count_parameters()

    def update_dropout(self, dropout) -> None:
        self.encoder.update_dropout(dropout)

    def forward(self, x: Tensor, input_lengths: Tensor) -> Tuple[Tensor, Tensor]:
        encoder_outputs, encoder_output_lengths = self.encoder(x, input_lengths)
        encoder_outputs = torch.mean(encoder_outputs, axis = 1)
        outputs = self.classification(encoder_outputs)
        return encoder_outputs, outputs, encoder_output_lengths

    def get_embedding(self, x: Tensor, input_lengths: Tensor) -> Tensor:
        with torch.no_grad():
            encoder_outputs, encoder_output_lengths = self.encoder(x, input_lengths)
            encoder_outputs = torch.mean(encoder_outputs, axis = 1)
        return encoder_outputs

# if __name__ == '__main__':
#     cuda = torch.cuda.is_available()
#     device = torch.device('cuda' if cuda else 'cpu')
#     batch_size, sequence_length, dim = 1024, 398, 80
#     inputs = torch.rand(batch_size, sequence_length, dim).to(device)
#     input_lengths = torch.LongTensor([398, 398, 398])
#     model = Conformer(num_classes = 663, input_dim = dim, encoder_dim = 512, num_encoder_layers = 6).to(device)
# #    encoder_outputs, outputs, output_lengths = model(inputs, input_lengths)
# #    encoder_outputs = model.get_embedding(inputs, input_lengths)
# #    print(encoder_outputs.shape)
# #    print(outputs.shape)
# #    print(output_lengths.shape)