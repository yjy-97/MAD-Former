import torch
import torch.nn as nn
from model.Transformer import TransformerModel
from model.PositionalEncoding import FixedPositionalEncoding, LearnedPositionalEncoding
from model.TDBFEN import BFEN
from model.PatchEmbedding import PatchEmbeddingBlock

class MADFormer(nn.Module):
    def __init__(
            self,
            img_dim,
            patch_dim,
            num_channels,
            embedding_dim,
            num_heads,
            num_layers,
            hidden_dim,
            dropout_rate=0.0,
            attn_dropout_rate=0.0,
            conv_patch_representation=True,
            positional_encoding_type="learned",
    ):
        super(MADFormer, self).__init__()

        assert embedding_dim % num_heads == 0
        assert img_dim % patch_dim == 0

        self.img_dim = img_dim
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.patch_dim = patch_dim
        self.num_channels = num_channels
        self.dropout_rate = dropout_rate
        self.attn_dropout_rate = attn_dropout_rate
        self.conv_patch_representation = conv_patch_representation

        self.num_patches = int((img_dim // patch_dim) ** 3)
        self.seq_length = self.num_patches
        self.flatten_dim = 128 * num_channels

        self.linear_encoding = nn.Linear(self.flatten_dim, self.embedding_dim)
        if positional_encoding_type == "learned":
            self.position_encoding = LearnedPositionalEncoding(
                self.seq_length, self.embedding_dim, self.seq_length
            )
        elif positional_encoding_type == "fixed":
            self.position_encoding = FixedPositionalEncoding(
                self.embedding_dim,
            )

        self.pe_dropout = nn.Dropout(p=self.dropout_rate)

        self.transformer = TransformerModel(
            embedding_dim,
            num_layers,
            num_heads,
            hidden_dim,

            self.dropout_rate,
            self.attn_dropout_rate,
        )
        self.pre_head_ln = nn.LayerNorm(embedding_dim)

        if self.conv_patch_representation:
            self.conv_x = nn.Conv3d(
                512,
                self.embedding_dim,
                kernel_size=3,
                stride=1,
                padding=1
            )

        self.ResNet = BFEN(num_classes=2)
        self.bn = nn.BatchNorm3d(1)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(32, 2)
        self.pe_8 = PatchEmbeddingBlock(in_channels=1, img_size=128, patch_size=8, hidden_size=32, num_heads=4,
                                        pos_embed="conv")
        self.pe_16 = PatchEmbeddingBlock(in_channels=1, img_size=128, patch_size=4, hidden_size=32, num_heads=4,
                                         pos_embed="conv")
        # self.fc_16 = nn.Linear(129, 17)

    def encode_8(self, x):
        if self.conv_patch_representation:
            x = self.ResNet(x)
            x = self.bn(x)
            x = self.relu(x)
            x_8 = self.pe_8(x)

        x_8 = self.pe_dropout(x_8)
        x_8, m = self.transformer(x_8)
        x_8 = self.pre_head_ln(x_8[0])
        return x_8, m

    def encode_16(self, x):
        if self.conv_patch_representation:
            x = self.ResNet(x)
            x = self.bn(x)
            x = self.relu(x)
            x_16 = self.pe_16(x)

        x_16 = self.pe_dropout(x_16)
        x_16, m = self.transformer(x_16)
        x_16 = self.pre_head_ln(x_16[0])
        return x_16, m

    def forward(self, x, auxillary_output_layers=[1, 2, 3, 4]):

        encoder_output_8, m_8 = self.encode_8(x)
        encoder_output_8 = encoder_output_8[:, 0]
        output_8 = self.fc(encoder_output_8)

        encoder_output_16, m_16 = self.encode_16(x)
        encoder_output_16 = encoder_output_16[:, 0]
        output_16 = self.fc(encoder_output_16)

        out= output_8 + output_16
        return out, m_8, m_16

    def _get_padding(self, padding_type, kernel_size):
        assert padding_type in ['SAME', 'VALID']
        if padding_type == 'SAME':
            _list = [(k - 1) // 2 for k in kernel_size]
            return tuple(_list)
        return tuple(0 for _ in kernel_size)

    def _reshape_output(self, x):
        x = x.view(
            x.size(0),
            int(self.img_dim / self.patch_dim),
            int(self.img_dim / self.patch_dim),
            int(self.img_dim / self.patch_dim),
            self.embedding_dim,
        )
        x = x.permute(0, 4, 1, 2, 3).contiguous()

        return x

class MF(MADFormer):
    def __init__(
            self,
            img_dim,
            patch_dim,
            num_channels,
            num_classes,
            embedding_dim,
            num_heads,
            num_layers,
            hidden_dim,
            dropout_rate=0.0,
            attn_dropout_rate=0.0,
            conv_patch_representation=True,
            positional_encoding_type="learned",
    ):
        super(MF, self).__init__(
            img_dim=img_dim,
            patch_dim=patch_dim,
            num_channels=num_channels,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate,
            attn_dropout_rate=attn_dropout_rate,
            conv_patch_representation=conv_patch_representation,
            positional_encoding_type=positional_encoding_type,
        )

        self.num_classes = num_classes

        self.Softmax = nn.Softmax(dim=1)

        self.endconv = nn.Conv3d(self.embedding_dim // 32, 4, kernel_size=1)


def MADFormer(dataset='mf', _conv_repr=True, _pe_type="learned"):
    if dataset.lower() == 'mf':
        img_dim = 32
        num_classes = 2

    num_channels = 1
    patch_dim = 8
    model = MF(
        img_dim,
        patch_dim,
        num_channels,
        num_classes,
        embedding_dim=32,
        num_heads=2,
        num_layers=2,
        hidden_dim=256,
        dropout_rate=0.1,
        attn_dropout_rate=0.1,
        conv_patch_representation=_conv_repr,
        positional_encoding_type=_pe_type,
    )

    return model


