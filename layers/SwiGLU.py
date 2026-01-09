from torch import nn

class SwiGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.w1 = nn.Linear(dim_in, dim_out, bias=False)
        self.w2 = nn.Linear(dim_in, dim_out, bias=False)
        self.proj = nn.Linear(dim_out, dim_in, bias=False)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.proj(self.act(self.w1(x)) * self.w2(x))
