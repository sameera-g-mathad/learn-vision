import torch
from torch import nn


class MLP(nn.Module):
    """
    Docstring for MLP

    Feed forward module of transformer block.
    Will be implementing SWiGlU, for a change.
    """

    def __init__(self, in_dim: int, hid_dim: int, out_dim: int) -> None:
        """
        Docstring for __init__

        :param in_dim: Inner dimension of fc1 and fc2.
        :type in_dim: int

        :param hid_dim: Hidden dimension for projection.
        :type hid_dim: int

        :param out_dim: Output dimension of the network.
        :type out_dim: int
        """
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hid_dim)
        self.fc2 = nn.Linear(in_dim, hid_dim)
        self.fc3 = nn.Linear(hid_dim, out_dim)

        self.silu = SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Docstring for forward

        :param x: Input tensor.
        :type x: torch.Tensor
        :return: Output tensor.
        :rtype: Tensor
        """
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        x = self.silu(x1) * x2
        return self.fc3(x)


class MultiHeadAttention(nn.Module):
    """
    Mutlihead Attention class. Used to convert
    regular tokens into context vectors where each
    token learns about other tokens in the context.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        qkv_bias: bool = False,
    ) -> None:
        """
        Docstring for __init__
        :param in_dim: The inner dimension (-1) of input sequence to match
                     the weights.
        :type in_dim: int

        :param out_dim: Intented output dimension of each token. Must be an even
                        number.
        :type out_dim: int

        :param num_heads: Number of multi-heads needed.
        :type num_heads: int

        :param dropout: Dropout value.
        :type dropout: float

        :param qkv_bias: Boolean whether bias is needed or not.
        :type qkv_bias: bool
        """
        super().__init__()

        # out_dim and num_heads should be even.
        assert out_dim % 2 == 0, "out_dim must be a even number."

        assert (
            out_dim % num_heads
        ), "Embedding dimension must be divisible by head dimension."

        # dropout must be between 0 and 1 inclusive.
        assert (
            0.0 <= dropout <= 1.0
        ), "Dropout should be in the range of (0, 1) inclusive."

        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads

        # weights for queries, keys, values.
        self.wq = nn.Linear(in_dim, out_dim, bias=qkv_bias)
        self.wk = nn.Linear(in_dim, out_dim, bias=qkv_bias)
        self.wv = nn.Linear(in_dim, out_dim, bias=qkv_bias)

        self.proj = nn.Linear(out_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Docstring for forward
        Method to perform multi-head attention.

        :param x: Input tensor that has no information about other
                  tokens in the sequence..
        :type x: torch.Tensor.

        :returns: Context vectors for the input passed.
        :rtype: torch.Tensor
        """
        b, context_len, _ = x.shape  # [b, context, in_dim]

        # project
        queries = self.wq(x)
        keys = self.wk(x)
        values = self.wv(x)

        # view the outdim with multiple heads.

        # [batch_size, num_heads, context_len, head_dim]
        queries = queries.view(b, self.num_heads, context_len, self.head_dim)

        # [batch_size, num_heads, context_len, head_dim]
        keys = keys.view(b, self.num_heads, context_len, self.head_dim)

        # [batch_size, num_heads, context_len, head_dim]
        values = values.view(b, self.num_heads, context_len, self.head_dim)

        # calculate attention scores
        # (qk.T) / √head_dim
        attn_scores = (queries @ keys.transpose(-2, -1)) / self.head_dim**0.5

        # no causal attention here for ViT.
        # take softmax to calculate attention weights.
        attn_weights = torch.softmax(attn_scores, dim=-1)

        # apply dropout
        attn_weights = self.dropout(attn_weights)

        # calculate attention weights.
        context_vec = attn_weights @ values

        # (b, context, out_dim)
        # out_dim = head_dim * num_heads
        context_vec = context_vec.view(b, context_len, self.head_dim * self.num_heads)

        # output projection of context vectors.
        context_vec = self.proj(context_vec)

        return context_vec


class RSNorm(nn.Module):
    """
    Docstring for RSNorm
    """

    def __init__(self, emb_dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(emb_dim)).float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Docstring for forward

        :param x: Input tensor to apply normalization on.
        :type x: torch.Tensor
        :return: Normalized tensor.
        :rtype: Tensor
        """
        x_mean = torch.pow(x, 2).mean(dim=-1, keepdim=True)
        x_norm = x * torch.rsqrt(self.eps * x_mean)
        return (x_norm * self.weight).to(x.dtype)


class SiLU(nn.Module):
    """
    Docstring for SiLU

    Implementation of Sigmoid Linear Unit.
    Silu(x) = x * sigmoid(x)
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Docstring for forward

        :param x: Input tensor.
        :type x: torch.Tensor
        :return: Activated tensor.
        :rtype: Tensor
        """
        return x * torch.sigmoid(x)
