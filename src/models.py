from torch import Tensor, nn
from jaxtyping import Float


class OnlyNet(nn.Module):
    def __init__(self, dim: int = 1280):
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
