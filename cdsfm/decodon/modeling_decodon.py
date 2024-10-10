from typing import Optional, Tuple
from dataclasses import dataclass
import torch
import torch.nn as nn

from transformers.modeling_outputs import (
    SequenceClassifierOutput,
)

from typing import Optional, Tuple

import torch
import torch.utils.checkpoint
from torch import nn

from dataclasses import dataclass
from transformers.activations import ACT2FN, ACT2CLS
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers.modeling_outputs import BaseModelOutputWithPast, BaseModelOutput, CausalLMOutputWithPast
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
import xformers.ops as xops

from collections import OrderedDict

logger = logging.get_logger(__name__)

import torch
import torch.nn as nn
from einops import rearrange, einsum
from transformers.pytorch_utils import Conv1D


import torch
from torch.amp import autocast
from torch import nn, einsum, Tensor

from einops import rearrange, repeat
from typing import Optional, Union

from .configuration_decodon import DeCodonConfig

logger = logging.get_logger(__name__)


def rotate_half(x):
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")


@autocast(device_type="cuda", enabled=False)
def apply_rotary_emb(freqs, t, start_index=0, scale=1.0):
    """
    Applies rotary embeddings to a tensor.

    Parameters
    ----------
    freqs : Tensor
        The frequencies to apply to the tensor: (seq_len, dim)
    t : Tensor
        The tensor to apply the rotary embeddings to: (..., seq_len, n_heads, dim)
    start_index : int
        The starting index to apply the rotary embeddings. (default: 0)
    scale : float
        The scale to apply to the rotary embeddings. (default: 1.0)

    Returns
    -------
    Tensor
        The tensor with the rotary embeddings applied.: (..., seq_len, n_heads, dim)

    """
    # if t.ndim == 3:
    #     seq_len = t.shape[seq_dim]
    #     freqs = freqs[-seq_len:].to(t)

    rot_dim = freqs.shape[-1]
    end_index = start_index + rot_dim

    assert (
        rot_dim <= t.shape[-1]
    ), f"feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}"

    t_left, t, t_right = (
        t[..., :start_index],
        t[..., start_index:end_index],
        t[..., end_index:],
    )
    if isinstance(scale, float):
        scale = torch.tensor(scale, device=t.device, dtype=t.dtype)

    t = (t * freqs.cos() * scale) + (rotate_half(t) * freqs.sin() * scale)
    return torch.cat((t_left, t, t_right), dim=-1)


# learned rotation helpers
def apply_learned_rotations(rotations, t, start_index=0, freq_ranges=None):
    if freq_ranges is not None:
        rotations = einsum("..., f -> ... f", rotations, freq_ranges)
        rotations = rearrange(rotations, "... r f -> ... (r f)")

    rotations = repeat(rotations, "... n -> ... (n r)", r=2)
    return apply_rotary_emb(rotations, t, start_index=start_index)


"""
Inspired from https://github.com/lucidrains/rotary-embedding-torch
"""

class RotaryEmbedding(nn.Module):
    """
    Rotary Embeddings Implemenetation inspired by https://github.com/lucidrains/rotary-embedding-torch.

    Rotary Positional Embeddings (RoPE) encode position information of tokens with a
    rotation matrix that naturally incorporates explicit relative position dependency.

    Parameters
    ----------
    emb_dim : int
        Embedding dimension. Usually set to the dim of each head in the attention module.
    freqs : Optional[Tensor]
        Custom frequencies to apply to query/key tensors. (default: None)
    theta : float
        Base constant used for computing rotation angles.
    learned_freq : bool (default: False)
        Whether to learn the frequencies.
    use_xpos : bool (default: False)
        Whether to employ XPos technique for resolving length extrapolation issue.
        NOTE: This can only be enabled for autoregressive models like GPT.
    xpos_scale_base : int (default: 512)
        The base for the scale factor used in XPos technique.
    interpolate_factor : float (default: 1.0)
        Length interpolation factor for extending context length of the pretrained model.
        Final model's context length = pretrained_model_context_length * interpolate_factor.

    theta_rescale_factor : float (default: 1.0)
        The factor to rescale the theta.

    cache_if_possible : bool (default: True)
        Whether to cache the frequencies/scales if possible.

    """

    def __init__(
        self,
        emb_dim,
        freqs: Optional[Tensor] = None,
        theta=1e4,
        learned_freq=False,
        use_xpos=False,
        xpos_scale_base=512,
        interpolate_factor=1.0,
        theta_rescale_factor=1.0,
        cache_if_possible=True,
    ):
        super().__init__()
        # proposed by reddit user bloc97, to rescale rotary embeddings to longer sequence length without fine-tuning
        # has some connection to NTK literature
        # https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/

        theta *= theta_rescale_factor ** (emb_dim / (emb_dim - 2))

        if freqs is None:
            freqs = 1.0 / (
                theta
                ** (torch.arange(0, emb_dim, 2)[: (emb_dim // 2)].float() / emb_dim)
            )
            # freqs = torch.ones(num_freqs).float()

        self.cache_if_possible = cache_if_possible

        self.register_buffer("cached_freqs", None, persistent=False)
        self.register_buffer("cached_scales", None, persistent=False)

        self.freqs = nn.Parameter(freqs, requires_grad=learned_freq)

        self.learned_freq = learned_freq

        # interpolation factors

        assert interpolate_factor >= 1.0
        self.interpolate_factor = interpolate_factor

        # xpos
        self.use_xpos = use_xpos
        if not use_xpos:
            self.register_buffer("scale", None, persistent=False)
            return

        scale = (torch.arange(0, emb_dim, 2) + 0.4 * emb_dim) / (1.4 * emb_dim)
        self.scale_base = xpos_scale_base
        self.register_buffer("scale", scale, persistent=False)

    @property
    def device(self):
        return self.freqs.device

    def rotate_queries_or_keys(self, t, offset=0, freq_seq_len=None, scale=None):
        """
        Parameters
        ----------
        t : Tensor
            tensor to rotate: (batch_size, seq_len, num_heads, head_dim)
        """
        seq_len = t.shape[1]
        assert (
            not self.use_xpos or scale is not None
        ), "you must use `.rotate_queries_and_keys` method instead and pass in both queries and keys, for length extrapolatable rotary embeddings"

        if freq_seq_len is not None:
            assert freq_seq_len >= seq_len
            seq_len = freq_seq_len

        seq = (
            torch.arange(seq_len, device=t.device, dtype=t.dtype) + offset
        ) / self.interpolate_factor

        freqs = self.forward(
            seq,
            seq_len=seq_len,
            offset=offset,
        ).to(t.dtype)

        freqs = rearrange(freqs, "n d -> n 1 d")

        if scale is not None:
            scale = rearrange(scale, "n d -> n 1 d")

        if scale is None:
            scale = torch.tensor(1.0, device=t.device, dtype=t.dtype)

        return apply_rotary_emb(freqs, t, scale=scale)

    def rotate_queries_and_keys(self, q, k):
        """
        Parameters
        ----------
        q : Tensor
            queries tensor: (batch_size, seq_len, num_heads, head_dim)
        k : Tensor
            keys tensor: (batch_size, seq_len, num_heads, head_dim)
        """
        assert self.use_xpos
        seq_len = q.shape[-3]

        seq = (
            torch.arange(seq_len, device=q.device, dtype=q.dtype)
        ) / self.interpolate_factor

        freqs = self.forward(seq, seq_len=seq_len)
        scale = self.get_scale(seq, seq_len=seq_len)

        freqs = rearrange(freqs, "n d -> n 1 d")
        scale = rearrange(scale, "n d -> n 1 d")

        rotated_q = apply_rotary_emb(freqs, q, scale=scale)
        rotated_k = apply_rotary_emb(freqs, k, scale=scale**-1)

        rotated_q = rotated_q.type(q.dtype)
        rotated_k = rotated_k.type(k.dtype)

        return rotated_q, rotated_k

    def get_scale(self, t: Tensor, seq_len: Optional[int] = None, offset=0):
        assert self.use_xpos

        should_cache = self.cache_if_possible and seq_len is not None

        if (
            should_cache
            and self.cached_scales is not None
            and (seq_len + offset) <= self.cached_scales.shape[0]
        ):
            return self.cached_scales[offset : (offset + seq_len)]

        scale = 1.0
        if self.use_xpos:
            power = (t - len(t) // 2) / self.scale_base
            scale = self.scale ** rearrange(power, "n -> n 1")
            scale = torch.cat((scale, scale), dim=-1)

        if should_cache:
            self.register_buffer("cached_scales", scale, persistent=False)

        return scale

    def rotate_queries_with_cached_keys(self, q, k, offset=0):
        q_len, k_len = q.shape[1], k.shape[1]
        assert q_len <= k_len

        rotated_q, rotated_k = self.rotate_queries_and_keys(q, k)

        rotated_q = rotated_q[:, -1:, ...]

        return rotated_q, rotated_k

        seq = (
            torch.arange(k_len, device=q.device, dtype=q.dtype)
        ) / self.interpolate_factor

        if self.use_xpos:
            q_scale = self.get_scale(seq[-q_len:]).to(q.dtype)
            k_scale = self.get_scale(seq).to(k.dtype)

        else:
            k_scale = 1.0
            q_scale = 1.0

        rotated_q = self.rotate_queries_or_keys(
            q, scale=q_scale, offset=k_len - q_len + offset
        )
        rotated_k = self.rotate_queries_or_keys(k, scale=k_scale**-1)

        return rotated_q, rotated_k

    @autocast(device_type="cuda", enabled=False)
    def forward(self, t: Tensor, seq_len=None, offset=0):
        should_cache = (
            self.cache_if_possible and not self.learned_freq and seq_len is not None
        )

        if (
            should_cache
            and self.cached_freqs is not None
            and (offset + seq_len) <= self.cached_freqs.shape[0]
        ):
            return self.cached_freqs[offset : (offset + seq_len)].detach()

        freqs = self.freqs

        freqs = einsum("..., f -> ... f", t, freqs)
        freqs = repeat(freqs, "... n -> ... (n r)", r=2)

        if should_cache:
            self.register_buffer("cached_freqs", freqs.detach(), persistent=False)

        return freqs



class MultiHeadedSelfAttention(nn.Module):
    """
    Multi-Headed Self Attention module supported with Flash Attention and Rotary Embeddings.

    Parameters
    ----------
    q_input_dim: int
        The input dimension of the query tensor.
    kv_input_dim: int
        The input dimension of the key and value tensors.
    qk_proj_dim: int
        The projected dimension of the query and key tensors.
    v_proj_dim: int
        The projected dimension of the value tensors.
    num_heads: int
        Number of attention heads.
    dropout: float
        Dropout rate to apply to the attention scores.
    projection_layer: str
        The type of projection layer to use. Either 'linear' or 'conv'.
        Basically both are linear projections, but 'conv' uses Conv1D layer as proposed in the original GPT2 paper.
    use_flash_attn: bool
        Whether to use Flash Attention or not. If True, Flash Attention will be used.
        NOTE: Flash Attention is required to be installed.
    use_rotary_emb: bool
        Whether to use Rotary Embeddings or not.
    rotary_theta: int
        The base for the geometric progression used to compute the rotation angles.
    rotary_use_xpos: bool
        Whether to use XPos technique for resolving length extrapolation issue.
        NOTE: This can only be enabled for autoregressive models like GPT.
    """

    def __init__(
        self,
        q_input_dim,
        kv_input_dim,
        qk_proj_dim,
        v_proj_dim,
        num_heads,
        dropout: float = 0.0,
        projection_layer: str = "linear",
        use_flash_attn: bool = True,
        use_rotary_emb: bool = False,
        rotary_theta: int = 1e4,
        rotary_use_xpos: bool = False,
        is_cross_attention: bool = False,
        **kwargs,
    ):
        super().__init__()
        assert (
            qk_proj_dim % num_heads == 0
        ), "qk_proj_dim must be divisible by num_heads"
        assert v_proj_dim % num_heads == 0, "v_proj_dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.dropout_rate = dropout
        self.projection_layer = projection_layer
        self.use_rotary_emb = use_rotary_emb
        self.is_cross_attention = is_cross_attention

        if use_flash_attn and not is_cross_attention:
            try:
                from flash_attn import flash_attn_qkvpacked_func

                self.use_flash_attn = True
                self.flashattn_fn = flash_attn_qkvpacked_func
            except ImportError:
                print("flash_attn not installed, reverting to default attention")
                self.use_flash_attn = False
                self.flashattn_fn = None
        else:
            self.use_flash_attn = False
            self.flashattn_fn = None

        if self.projection_layer == "linear":
            self.query = nn.Linear(q_input_dim, qk_proj_dim)
            self.key = nn.Linear(kv_input_dim, qk_proj_dim)
            self.value = nn.Linear(kv_input_dim, v_proj_dim)
        elif self.projection_layer == "conv":
            self.query = Conv1D(qk_proj_dim, q_input_dim)
            self.key = Conv1D(qk_proj_dim, kv_input_dim)
            self.value = Conv1D(v_proj_dim, kv_input_dim)
        else:
            raise ValueError(
                f"projection_layer must be either 'linear' or 'conv', got {projection_layer}"
            )

        if self.use_rotary_emb:
            self.rotary_emb = RotaryEmbedding(
                emb_dim=qk_proj_dim // num_heads // 2,
                theta=rotary_theta,
                use_xpos=rotary_use_xpos,
            )

        self.dr_rate = dropout
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x_q,
        x_kv,
        is_causal=False,
        attention_bias=None,
        attention_mask=None,
        output_attentions=False,
        query=None,
        key=None,
        value=None,
        use_cache=False,
    ):
        """
        Applies a classical self attention operation.

        Parameters
        ----------
        x_q: torch.Tensor
            The query tensor of shape (batch_size, query_seq_len, emb_dim)
        x_kv: torch.Tensor
            The key/value tensor of shape (batch_size, kv_seq_len, emb_dim)
        attention_bias: torch.Tensor
            The attention bias to apply to the attention scores. (default: None)
        attention_mask: torch.Tensor
            The attention mask to apply to the attention scores. Shape: (batch_size, q_len, kv_seq_len)
        """
        assert (x_q is not None and x_kv is not None) or (
            query is not None and key is not None and value is not None
        ), "Either x_q and x_kv or query, key and value must be provided"

        past_memory_provided = (
            query is not None and key is not None and value is not None
        )

        if query is None:
            q_len = x_q.size(1)
            k_len = x_kv.size(1)

            query = self.query(x_q)
            key = self.key(x_kv)
            value = self.value(x_kv)

        else:
            q_len = query.size(1)
            k_len = key.size(1)

        if use_cache:
            cache = (key.clone(), value.clone(), query.clone())

        q = rearrange(query, "b q (h d) -> b q h d", h=self.num_heads)
        k = rearrange(key, "b k (h d) -> b k h d", h=self.num_heads)
        v = rearrange(value, "b v (h d) -> b v h d", h=self.num_heads)

        if self.use_rotary_emb:
            if use_cache and past_memory_provided:
                q, k = self.rotary_emb.rotate_queries_with_cached_keys(q, k)
            if self.rotary_emb.use_xpos:
                q, k = self.rotary_emb.rotate_queries_and_keys(q, k)
            else:
                q = self.rotary_emb.rotate_queries_or_keys(q)
                k = self.rotary_emb.rotate_queries_or_keys(k)

        if (
            self.use_flash_attn
            and not use_cache
            and not output_attentions
            and attention_bias is None
        ):
            qkv = torch.stack([q, k, v], dim=2).to(torch.bfloat16)
            x = self.flashattn_fn(
                qkv=qkv,
                dropout_p=self.dropout_rate if self.training else 0.0,
                causal=is_causal,
                deterministic=False,
                return_attn_probs=False,
            )

            x = x.to(x_q.dtype)
        elif self.use_flash_attn and not output_attentions:
            attn_bias = xops.LowerTriangularMask() if is_causal else attention_bias

            if attention_mask is not None:
                if attn_bias is None:
                    attn_bias = attention_mask
                else:
                    if isinstance(attn_bias, torch.Tensor):
                        attn_bias = attn_bias + attention_mask
                    else:
                        attn_bias.add_bias(bias=attention_mask)

                        attn_bias = attn_bias.materialize(
                            shape=(q_len, k_len),
                            device=q.device,
                            dtype=q.dtype,
                        )
            else:
                if isinstance(attn_bias, torch.Tensor) and len(attn_bias.shape) == 3:
                    attn_bias = (
                        attn_bias.unsqueeze(1)
                        .expand(-1, self.num_heads, -1, -1)
                        .float()
                    )  # (batch_size, num_heads, q_len, k_len)
                else:
                    attn_bias = attn_bias.materialize(
                        shape=(q_len, k_len),
                        device=q.device,
                        dtype=q.dtype,
                    )

            if isinstance(attn_bias, xops.LowerTriangularMask):
                attn_bias = attn_bias.materialize(
                    shape=(q_len, k_len),
                    device=q.device,
                    dtype=q.dtype,
                )

                # print(attention_mask.shape, attn_bias.shape)
                # print(attn_bias[0, 0, 0, :])

            need_adjustment = False
            if attn_bias.shape[-2] % 8 != 0:
                nearest_multiple_q = 8 * (1 + attn_bias.shape[-2] // 8)
                need_adjustment = True
            else:
                nearest_multiple_q = attn_bias.shape[-2]

            if attn_bias.shape[-1] % 8 != 0:
                nearest_multiple_k = 8 * (1 + attn_bias.shape[-1] // 8)
                need_adjustment = True
            else:
                nearest_multiple_k = attn_bias.shape[-1]

            if need_adjustment:
                new_attn_bias = torch.zeros(
                    attn_bias.shape[0],
                    attn_bias.shape[1],
                    nearest_multiple_q,
                    nearest_multiple_k,
                ).to(attn_bias.device)
                new_attn_bias[:, :, : attn_bias.shape[-2], : attn_bias.shape[-1]] = (
                    attn_bias
                )

                x = xops.memory_efficient_attention(
                    query=q,
                    key=k,
                    value=v,
                    op=None,
                    attn_bias=new_attn_bias[:, :, :q_len, :k_len],
                    p=self.dr_rate,
                )
            else:
                attn_bias = attn_bias.to(q.dtype)
                attn_bias = attn_bias.repeat(1, self.num_heads, 1, 1)
                x = xops.memory_efficient_attention(
                    query=q,
                    key=k,
                    value=v,
                    op=None,
                    attn_bias=attn_bias,
                    p=self.dr_rate,
                )
            # x: (batch_size, query_seq_len, n_head, head_dim)
        else:
            # if output_attentions:
            attention_scores = einsum(q, k, "b q h d, b k h d -> b h q k")
            attention_scores = attention_scores / (q.size(-1) ** 0.5)

            if attention_bias is not None:
                attn_bias = attention_bias.unsqueeze(1).expand(
                    -1, self.num_heads, -1, -1
                )
            # elif is_causal:
            #     attn_bias = xops.LowerTriangularMask().materialize(
            #         shape=attention_scores.shape, device=attention_scores.device
            #     )
            else:
                attn_bias = None

            if attention_mask is not None:
                if attn_bias is None:
                    attn_bias = attention_mask
                else:
                    attn_bias = attn_bias + attention_mask

            attention_scores = attention_scores + attn_bias

            attention_probs = attention_scores.softmax(dim=-1)
            attention_probs = self.dropout(attention_probs)

            x = einsum(attention_probs, v, "b h q k, b v h d -> b q h d")

        x = rearrange(x, "b q h d -> b q (h d)", h=self.num_heads)

        if use_cache:
            if output_attentions:
                return x, attention_probs, cache
            else:
                return x, None, cache
        else:
            if output_attentions:
                return x, attention_probs
            else:
                return x, None

class DeCodonPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    base_model_prefix = "decodon"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """MAGNETO Initialize the weights"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight, gain=self.config.gamma_init)
            if module.bias is not None:
                module.bias.data.zero_()

        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, DeCodonLayer):
            module.gradient_checkpointing = value


class DeCodonEmbeddings(nn.Module):
    """
    DeCodon Embeddings

    Word, position and token type embeddings for DeCodon.
    """

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )

        self.ln = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.position_embedding_type = getattr(
            config, "position_embedding_type", "absolute"
        )

        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings).expand((1, -1)),
            persistent=False,
        )

        self.register_buffer(
            "token_type_ids",
            torch.zeros(self.position_ids.size(), dtype=torch.long),
            persistent=False,
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values_length: int = 0,
    ) -> torch.Tensor:

        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[
                :, past_key_values_length : seq_length + past_key_values_length
            ]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(
                    input_shape[0], seq_length
                )
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(
                    input_shape, dtype=torch.long, device=self.position_ids.device
                )

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings

        # embeddings = self.ln(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class DeCodonAttention(nn.Module):
    """
    DeCodon Attention Layer

    This module supports self-attention and dilated attention with Rotary Positional Embeddings (RoPE).
    """

    def __init__(self, config):
        super().__init__()

        self.pre_layer_norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.post_attn_dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.post_layer_norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.self_attention = MultiHeadedSelfAttention(
            q_input_dim=config.hidden_size,
            kv_input_dim=config.hidden_size,
            qk_proj_dim=config.hidden_size,
            v_proj_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_probs_dropout_prob,
            projection_layer="conv",
            use_flash_attn=config.use_flash_attn,
            use_rotary_emb=config.use_rotary_emb,
            rotary_theta=config.rotary_theta,
            rotary_use_xpos=True,
        )

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        attention_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        past_key_values: Optional[Tuple[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:

        attn_input = self.pre_layer_norm(hidden_states)

        if past_key_values is not None:
            query = self.self_attention.query(attn_input)
            key = self.self_attention.key(attn_input)
            value = self.self_attention.value(attn_input)

            past_key, past_value, past_query = past_key_values

            # past_new_query = query[:, :-1, :]
            # past_new_key = key[:, :-1, :]
            # past_new_value = value[:, :-1, :]

            # print(
            #     (past_new_query[0] != past_query[0]).sum(),
            #     past_new_query.size(),
            #     past_new_query[past_new_query != past_query].cpu().numpy(),
            #     past_query[past_new_query != past_query].cpu().numpy(),
            #     past_query.sum().item(),
            # )
            # print(
            #     (past_new_key[0] == past_key[0]).sum(),
            #     past_new_key.size(),
            #     # past_new_key[0, 0, :1024],
            #     # past_key[0, 0, :1024],
            #     past_new_key[past_new_key != past_key].cpu().numpy(),
            #     past_key[past_new_key != past_key].cpu().numpy(),
            #     past_key.sum().item(),
            # )

            # print(
            #     (past_new_value[0] == past_value[0]).sum(),
            #     past_new_value.size(),
            #     # past_new_value[0, 0, :1024],
            #     # past_value[0, 0, :1024],
            #     past_new_value[past_new_value != past_value].cpu().numpy(),
            #     past_value[past_new_value != past_value].cpu().numpy(),
            #     past_value.sum().item(),
            # )

            # print(query.shape, key.shape, value.shape)
            # print(past_query.shape, past_key.shape, past_value.shape)

            key = torch.cat(
                (past_key, key), dim=1
            )  # (batch_size, seq_len, hidden_size)
            value = torch.cat(
                (past_value, value), dim=1
            )  # (batch_size, seq_len, hidden_size)
            query = torch.cat((past_query, query), dim=1)

            # print(query.shape, key.shape, value.shape)
            # print()

            attn_outputs = self.self_attention(
                x_q=None,
                x_kv=None,
                query=query,
                key=key,
                value=value,
                is_causal=True,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                use_cache=use_cache,
                attention_bias=None,
            )
        else:
            attn_outputs = self.self_attention(
                x_q=attn_input,
                x_kv=attn_input,
                is_causal=True,
                attention_bias=None,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        attn_output = attn_outputs[0]
        attn_output = self.post_layer_norm(attn_output)
        attn_output = self.post_attn_dense(attn_output)
        attn_output = self.dropout(attn_output)
        attn_output = hidden_states + attn_output

        return (attn_output,) + attn_outputs[1:]


class DeCodonFFN(nn.Module):
    """
    DeCodon Position-wise Feed-Forward Network
    """

    def __init__(self, config):
        super().__init__()
        embed_dim = config.hidden_size
        self.pre_layer_norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.intermediate_dense = Conv1D(config.intermediate_size, embed_dim)
        self.post_layer_norm = nn.LayerNorm(
            config.intermediate_size, eps=config.layer_norm_eps
        )
        self.post_dense = Conv1D(embed_dim, config.intermediate_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(
        self, hidden_states: Optional[Tuple[torch.FloatTensor]]
    ) -> torch.FloatTensor:
        hidden_states = self.pre_layer_norm(hidden_states)
        hidden_states = self.intermediate_dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.post_layer_norm(hidden_states)
        hidden_states = self.post_dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class DeCodonLayer(nn.Module):
    """
    DeCodon (Decoder) Layer consists of an attention layer and a position-wise feed-forward network.
    """

    def __init__(self, config):
        super().__init__()
        self.attention = DeCodonAttention(config)
        self.output = DeCodonFFN(config)

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        attention_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        past_key_values: Optional[Tuple[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = False,
    ) -> Union[
        Tuple[torch.Tensor],
        Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]],
    ]:
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            output_attentions=output_attentions,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        attention_output = self_attention_outputs[0]

        outputs = self_attention_outputs[
            1:
        ]  # add self attentions if we output attention weights

        layer_output = self.output(attention_output)
        outputs = (layer_output,) + outputs

        return outputs


class DeCodonStack(nn.Module):
    """
    DeCodon Stack consists of multiple DeCodon layers.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.blocks = nn.ModuleList(
            [DeCodonLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[torch.FloatTensor]] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
        use_cache: Optional[bool] = False,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutput]:

        if past_key_values is None:
            past_key_values = [None] * len(self.blocks)
            past_length = 0
        else:
            past_length = past_key_values[0][0].size(-2)

        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        presents = () if use_cache else None
        for i, (block, past_key_value) in enumerate(zip(self.blocks, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            block_outputs = block(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                past_key_values=past_key_value,
                use_cache=use_cache,
            )

            hidden_states = block_outputs[0]

            if use_cache:
                presents = presents + (block_outputs[2],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (block_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    presents,
                    all_hidden_states,
                    all_self_attentions,
                ]
                if v is not None
            )

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class DeCodonModule(DeCodonPreTrainedModel):
    """
    The DeCodon Module (Decoder only) without any task-specific head on top.
    """
 
    def __init__(self, config):
        super().__init__(config)

        self.embeddings = DeCodonEmbeddings(config)
        self.decoder = DeCodonStack(config)
        self.ln_f = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()
    
    def set_input_embeddings(self, new_embeddings):
        self.embeddings.word_embeddings = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[torch.FloatTensor]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        use_cache: Optional[bool] = False,
    ) -> Union[Tuple, BaseModelOutput]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if past_key_values is not None:
            past_length = past_key_values[0][0].size(-2)
        else:
            past_length = 0

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(
                    batch_size, seq_length
                )
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(
                    input_shape, dtype=torch.long, device=device
                )

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        # extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
        #     attention_mask, input_shape
        # )
        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )

        extended_attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask=attention_mask,
            input_shape=(batch_size, input_shape[-1]),
            inputs_embeds=embedding_output,
            past_key_values_length=past_length,
        )
        # extended_attention_mask = attention_mask

        decoder_outputs = self.decoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            past_key_values=past_key_values,
            return_dict=return_dict,
            use_cache=use_cache,
        )

        sequence_output = decoder_outputs[0]

        if not return_dict:
            return (sequence_output,) + decoder_outputs[1:]

        return BaseModelOutputWithPast(
            last_hidden_state=sequence_output,
            past_key_values=decoder_outputs.past_key_values,
            hidden_states=decoder_outputs.hidden_states,
            attentions=decoder_outputs.attentions,
        )


@dataclass
class DeCodonForPreTrainingOutput(CausalLMOutputWithPast):
    """
    Output type of [`BERTransForPreTraining`].

    Args:
        loss (*optional*, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`):
            Total loss as the sum of the masked language modeling loss and the next sequence prediction
            (classification) loss.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        org_logits (`torch.FloatTensor` of shape `(batch_size, 1)`):
            Prediction scores for organism classification (scores for each organism label before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class DeCodon(DeCodonPreTrainedModel):
    config_class = DeCodonConfig
    _tied_weights_keys = []

    def __init__(self, config):
        super().__init__(config)

        self.gpt = DeCodonModule(config)

        # causal language modeling head
        if config.lm_type == "gpt":
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            DeCodon._tied_weights_keys.append("lm_head.weight")
        else:
            self.lm_head = nn.Sequential(
                OrderedDict(
                    [
                        ("dropout", nn.Dropout(config.hidden_dropout_prob)),
                        (
                            "transform",
                            nn.Linear(config.hidden_size, config.hidden_size),
                        ),
                        ("act", nn.ReLU()),
                        (
                            "norm",
                            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
                        ),
                        (
                            "pred",
                            nn.Linear(
                                config.hidden_size, config.vocab_size, bias=False
                            ),
                        ),
                    ]
                )
            )
            DeCodon._tied_weights_keys.append("lm_head.pred.weight")

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.gpt.embeddings.word_embeddings

    def get_output_embeddings(self):
        return (
            self.lm_head.pred.weight
            if isinstance(self.lm_head, nn.Sequential)
            else self.lm_head.weight if self.config.lm_type == "gpt" else None
        )

    def set_output_embeddings(self, new_embeddings):
        if isinstance(self.lm_head, nn.Sequential):
            self.lm_head.pred.weight = new_embeddings
        else:
            self.lm_head.weight = new_embeddings

    def prepare_inputs_for_generation(
        self, input_ids, inputs_embeds=None, past_key_values=None, **kwargs
    ):
        token_type_ids = kwargs.get("token_type_ids", None)
        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)
        use_cache = kwargs.get("use_cache", True)

        if past_key_values is not None and use_cache:
            past_length = past_key_values[0][0].shape[1]

            if input_ids.shape[1] > past_length:
                remove_prefix_len = past_length
            else:
                remove_prefix_len = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_len:]

            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, remove_prefix_len:]

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
        else:
            position_ids = None

        if inputs_embeds is not None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache", True),
            }
        )

        return model_inputs

    @staticmethod
    def _reorder_cache(
        past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """
        return tuple(
            tuple(
                past_state.index_select(0, beam_idx.to(past_state.device))
                for past_state in layer_past
            )
            for layer_past in past_key_values
        )

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        organism: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Union[Tuple[torch.Tensor], DeCodonForPreTrainingOutput]:
        r"""
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
                config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked),
                the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
            organism (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                Organism labels
            kwargs (`Dict[str, any]`, optional, defaults to *{}*):
                Used to hide legacy arguments that have been deprecated.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, BertForPreTraining
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("bertrans-base")
        >>> model = BERTransForPreTraining.from_pretrained("bertrans-base")

        >>> inputs = tokenizer("AAAAGGGGGGCCCCCCTTTTT", return_tensors="pt")
        >>> outputs = model(**inputs)

        >>> prediction_logits = outputs.prediction_logits
        >>> organism_logits = outputs.organism_logits
        >>> biotype_logits = outputs.biotype_logits
        ```
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if input_ids is not None:
            batch_size, sequence_length = input_ids.shape[:2]
        else:
            batch_size, sequence_length = inputs_embeds.shape[:2]

        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (
                    torch.eq(input_ids, self.config.pad_token_id).long().argmax(-1) - 1
                ).to(input_ids.device)
            else:
                sequence_lengths = -1
                logger.warning(
                    f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                    "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
                )

        gpt_outputs = self.gpt(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            use_cache=use_cache,
        )

        hidden_states = gpt_outputs[0]  # (batch_size, sequence_length, hidden_size)
        lm_logits = self.lm_head(
            hidden_states
        )  # (batch_size, sequence_length, vocab_size)

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(lm_logits.device)
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            lm_loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            loss = lm_loss
        else:
            lm_loss = None

        if not return_dict:
            output = (lm_logits,) + gpt_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return DeCodonForPreTrainingOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=gpt_outputs.past_key_values,
            hidden_states=gpt_outputs.hidden_states,
            attentions=gpt_outputs.attentions,
        )
    
    def freeze(self, layer_indices: Optional[list] = None):
        if layer_indices is None or len(layer_indices) == 0:
            for param in self.gpt.parameters():
                param.requires_grad = False
        else:
            for param in self.gpt.embeddings.parameters():
                param.requires_grad = False

            if isinstance(layer_indices, int):
                layer_indices = [layer_indices]

            layer_indices = [i % len(self.gpt.decoder.blocks) for i in layer_indices]

            for i in range(len(self.gpt.decoder.blocks)):
                if i not in layer_indices:
                    for param in self.gpt.decoder.blocks[i].parameters():
                        param.requires_grad = False



class DeCodonForSequenceTask(DeCodonPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.gpt = DeCodonModule(config)

        if config.cls_type.lower() == "cls":
            layer_indices = config.layer_indices
            layer_indices = (
                []
                if layer_indices is None
                else (
                    [layer_indices] if isinstance(layer_indices, int) else layer_indices
                )
            )
            layer_indices = [i % len(self.gpt.decoder.blocks) for i in layer_indices]

            n_layers = len(layer_indices)
            self.layer_indices = layer_indices
            self.classifier = nn.Sequential(
                nn.LayerNorm(config.hidden_size * n_layers),
                nn.Linear(config.hidden_size * n_layers, config.hidden_size),
                ACT2CLS[config.cls_hidden_act](),
                nn.Dropout(config.cls_dropout_prob),
                nn.Linear(
                    config.hidden_size,
                    config.num_labels * config.num_tasks,
                ),
            )
        else:
            raise ValueError(f"Invalid cls_type: {config.cls_type}.")

        self.init_weights()

    def freeze(self, layers_idx: Optional[list] = None):
        if layers_idx is None or len(layers_idx) == 0:
            for param in self.gpt.parameters():
                param.requires_grad = False
        else:
            for param in self.gpt.embeddings.parameters():
                param.requires_grad = False

            if isinstance(layers_idx, int):
                layers_idx = [layers_idx]

            layers_idx = [i % self.config.num_hidden_layers for i in layers_idx]

            for i in range(self.config.num_hidden_layers):
                if i not in layers_idx:
                    for param in self.gpt.decoder.blocks[i].parameters():
                        param.requires_grad = False

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        target: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if input_ids is not None:
            batch_size, sequence_length = input_ids.shape[:2]
        else:
            batch_size, sequence_length = inputs_embeds.shape[:2]

        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (
                    torch.eq(input_ids, self.config.pad_token_id).long().argmax(-1) - 1
                ).to(
                    input_ids.device
                )  # (batch_size,)
            else:
                sequence_lengths = -1
                logger.warning(
                    f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                    "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
                )

        gpt_outputs = self.gpt(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )

        all_hidden_states = gpt_outputs.hidden_states

        if self.config.cls_type.lower() not in ["crossattention", "ca", "cls"]:
            logits, _ = self.classifier(all_hidden_states, attention_mask)
        elif self.config.cls_type.lower() in ["crossattention", "ca"]:
            bs, seq_len = input_ids.shape

            query_tasks = self.task_embeddings.weight  # (num_tasks, hidden_size)
            query_tasks = query_tasks.unsqueeze(0).expand(
                bs, -1, -1
            )  # (batch_size, num_tasks, hidden_size)

            cls_outputs = self.classifier(
                query_tasks,
                all_hidden_states,
                attention_mask,
                output_attentions=output_attentions,
            )  # (batch_size, num_tasks, num_labels)

            logits, ca = cls_outputs

            logits = logits.squeeze()
        elif self.config.cls_type.lower() == "cls":
            bs, seq_len = input_ids.shape
            # here we select latest token's hidden states as pooled output
            pooled_hidden_states = [
                h[torch.arange(bs, device=h.device), sequence_lengths - 1, :]
                for i, h in enumerate(all_hidden_states)
                if i in self.layer_indices
            ]
            pooled_output = torch.cat(
                pooled_hidden_states, dim=-1
            )  # (batch_size, hidden_size * n_layers)

            logits = self.classifier(pooled_output)

        loss = None
        if target is not None:
            if self.config.problem_type == "regression":
                logits = logits.view(-1, self.config.num_labels * self.config.num_tasks)
                target = target.view(-1, self.config.num_labels * self.config.num_tasks)

                mask = target != -500.0
                
                if self.config.loss_fn == "mse":
                    loss_fct = nn.MSELoss()
                    loss = loss_fct(logits[mask], target[mask])
                elif self.config.loss_fn == "mae":
                    loss_fct = nn.L1Loss()
                    loss = loss_fct(logits[mask], target[mask])
                elif self.config.loss_fn == "huber":
                    loss_fct = nn.SmoothL1Loss()
                    loss = loss_fct(logits[mask], target[mask])
                else:
                    raise ValueError(f"Invalid loss_fn: {self.config.loss_fn}.")
            else:
                loss_fct = nn.CrossEntropyLoss()

                logits = logits.view(-1, self.config.num_labels * self.config.num_tasks)
                target = target.view(
                    -1,
                )

                loss = loss_fct(logits, target)

        if not return_dict:
            output = (logits,) + gpt_outputs[2:]
            return ((loss,) + output) if loss is not None else output

        if output_attentions:
            if ca is not None:
                attentions = gpt_outputs.attentions + [ca]
            else:
                attentions = gpt_outputs.attentions
        else:
            attentions = None

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=pooled_output,
            attentions=attentions,
        )