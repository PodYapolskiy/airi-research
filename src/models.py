from typing import Literal

import torch
from torch import Tensor, nn
from jaxtyping import Float


class OnlyNet(nn.Module):
    def __init__(self, dim: int):
        super(OnlyNet, self).__init__()
        self.linear1 = nn.Linear(dim, 256)
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(128, 1)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, embedding):
        x: Float[Tensor, "batch dim"] = embedding  # noqa: F722
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.linear3(x)
        return x


class MetaNet(nn.Module):
    def __init__(self, dim: int = 13):
        super(MetaNet, self).__init__()
        self.linear1 = nn.Linear(dim, 256)
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(128, 1)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, embedding):
        x: Float[Tensor, "batch dim"] = embedding  # noqa: F722
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.linear3(x)
        return x


class VideoNet(nn.Module):
    def __init__(self, dim: int = 1280):
        super(VideoNet, self).__init__()
        self.linear1 = nn.Linear(dim, 256)
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(128, 1)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, embedding):
        x: Float[Tensor, "batch dim"] = embedding  # noqa: F722
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.linear3(x)
        return x


class AudioNet(nn.Module):
    def __init__(self, dim: int = 512):
        super(AudioNet, self).__init__()
        self.linear1 = nn.Linear(dim, 256)
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(128, 1)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, embedding):
        x: Float[Tensor, "batch dim"] = embedding  # noqa: F722
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.linear3(x)
        return x


class TextNet(nn.Module):
    def __init__(self, dim: int = 1280):
        super(TextNet, self).__init__()
        self.linear1 = nn.Linear(dim, 256)
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(128, 1)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, embedding):
        x: Float[Tensor, "batch dim"] = embedding  # noqa: F722
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.linear3(x)
        return x


class MultimodalNet(nn.Module):
    def __init__(self, dim: int):
        super(MultimodalNet, self).__init__()
        self.linear1 = nn.Linear(dim, 256)
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(128, 1)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, embedding):
        x: Float[Tensor, "batch dim"] = embedding  # noqa: F722
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.linear3(x)
        return x


class PerformanceNet(nn.Module):
    def __init__(
        self,
        with_meta: bool = False,
        with_video: bool = False,
        with_audio: bool = False,
        with_text: bool = False,
        meta_dim: int = 13,
        video_dim: int = 1280,
        audio_dim: int = 512,
        text_dim: int = 1024,
        fusion_strategy: Literal["early", "late"] = "late",
    ):
        super(PerformanceNet, self).__init__()
        self.modalities_count = 0
        self.modalities_dim = 0
        self.with_meta = with_meta
        self.with_video = with_video
        self.with_audio = with_audio
        self.with_text = with_text
        self.fusion_strategy = fusion_strategy

        if with_meta:
            self.modalities_count += 1
            self.modalities_dim += meta_dim
        if with_video:
            self.modalities_count += 1
            self.modalities_dim += video_dim
        if with_audio:
            self.modalities_count += 1
            self.modalities_dim += audio_dim
        if with_text:
            self.modalities_count += 1
            self.modalities_dim += text_dim

        if fusion_strategy == "early":
            self.net = MultimodalNet(self.modalities_dim)
        elif fusion_strategy == "late":
            self.meta_net = MetaNet(meta_dim) if with_meta else None
            self.video_net = VideoNet(video_dim) if with_video else None
            self.audio_net = AudioNet(audio_dim) if with_audio else None
            self.text_net = TextNet(text_dim) if with_text else None
        else:
            raise NotImplementedError()

    def forward(
        self,
        meta_embedding: Float[Tensor, "..."] | None,
        video_embedding: Float[Tensor, "..."] | None,
        audio_embedding: Float[Tensor, "..."] | None,
        text_embedding: Float[Tensor, "..."] | None,
    ):
        if self.fusion_strategy == "early":
            embeddings = []
            if self.with_meta:
                embeddings.append(meta_embedding)
            if self.with_video:
                embeddings.append(video_embedding)
            if self.with_audio:
                embeddings.append(audio_embedding)
            if self.with_text:
                embeddings.append(text_embedding)
            x_concat = torch.concat(embeddings, dim=0)
            x = self.net(x_concat)

        elif self.fusion_strategy == "late":
            x_meta = self.meta_net(meta_embedding) if self.with_meta else 0
            x_video = self.video_net(video_embedding) if self.with_video else 0
            x_audio = self.audio_net(audio_embedding) if self.with_audio else 0
            x_text = self.text_net(text_embedding) if self.with_text else 0
            x = (x_meta + x_video + x_audio + x_text) / self.modalities_count

        return x


class PersonalityNet(PerformanceNet):
    def __init__(self, **kwargs):
        super(PersonalityNet, self).__init__(**kwargs)
