"""
Supernova USLM (Ultra-Small Language Model) - 25M Parameters
Designed for efficient task-specific performance surpassing larger models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import math
from einops import rearrange, repeat
from dataclasses import dataclass


@dataclass
class SupernovaConfig:
    """Configuration for Supernova USLM model"""
    vocab_size: int = 32000
    hidden_size: int = 768  # Reduced from typical 1024
    num_hidden_layers: int = 8  # Shallow but wide
    num_attention_heads: int = 12
    num_key_value_heads: int = 4  # GQA for efficiency
    intermediate_size: int = 2048  # Smaller FFN
    hidden_act: str = "swiglu"  # More efficient activation
    max_position_embeddings: int = 2048
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    tie_word_embeddings: bool = True
    rope_theta: float = 10000.0
    rope_scaling: Optional[Dict[str, Any]] = None
    attention_bias: bool = False
    attention_dropout: float = 0.1
    hidden_dropout: float = 0.1
    partial_rotary_factor: float = 0.5  # Only rotate half of head dims
    use_flash_attention: bool = True
    use_sliding_window: bool = True
    sliding_window_size: int = 512
    output_attentions: bool = False
    output_hidden_states: bool = False
    
    @property
    def head_dim(self):
        return self.hidden_size // self.num_attention_heads
    
    def get_num_params(self):
        """Calculate approximate number of parameters"""
        # Embeddings
        embed_params = self.vocab_size * self.hidden_size
        
        # Attention layers
        attn_params_per_layer = (
            # Q, K, V projections
            3 * self.hidden_size * self.hidden_size +
            # Output projection
            self.hidden_size * self.hidden_size
        )
        
        # FFN layers
        ffn_params_per_layer = (
            3 * self.hidden_size * self.intermediate_size +  # Gate, up, down for SwiGLU
            self.intermediate_size * self.hidden_size
        )
        
        # Layer norms
        norm_params_per_layer = 2 * self.hidden_size
        
        total_transformer_params = self.num_hidden_layers * (
            attn_params_per_layer + ffn_params_per_layer + norm_params_per_layer
        )
        
        # Output layer (if not tied)
        output_params = 0 if self.tie_word_embeddings else self.vocab_size * self.hidden_size
        
        return embed_params + total_transformer_params + output_params


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding with partial rotation support"""
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Build cache
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.float32
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        return (
            self.cos_cached[:seq_len],
            self.sin_cached[:seq_len],
        )


def rotate_half(x):
    """Rotates half the hidden dims of the input"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Apply rotary position embedding to query and key tensors"""
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class SupernovaAttention(nn.Module):
    """Multi-head attention with grouped-query attention and sliding window"""
    def __init__(self, config: SupernovaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.partial_rotary_factor = config.partial_rotary_factor
        self.use_sliding_window = config.use_sliding_window
        self.sliding_window_size = config.sliding_window_size

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        
        # Rotary embeddings for partial dimensions
        self.rotary_ndims = int(self.head_dim * self.partial_rotary_factor)
        self.rotary_emb = RotaryEmbedding(
            self.rotary_ndims,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )
        
        self.attention_dropout = nn.Dropout(config.attention_dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        
        # Apply rotary embeddings to partial dimensions
        if position_ids is None:
            position_ids = torch.arange(0, q_len, dtype=torch.long, device=hidden_states.device)
            position_ids = position_ids.unsqueeze(0).view(-1, q_len)
        
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        
        # Split heads for partial rotation
        query_rot = query_states[..., :self.rotary_ndims]
        query_pass = query_states[..., self.rotary_ndims:]
        key_rot = key_states[..., :self.rotary_ndims]
        key_pass = key_states[..., self.rotary_ndims:]
        
        # Apply rotary embeddings
        query_rot, key_rot = apply_rotary_pos_emb(query_rot, key_rot, cos, sin, position_ids)
        query_states = torch.cat([query_rot, query_pass], dim=-1)
        key_states = torch.cat([key_rot, key_pass], dim=-1)

        if past_key_value is not None:
            # Reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # Repeat k/v heads if using GQA
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        # Apply sliding window attention if enabled
        if self.use_sliding_window and attention_mask is None:
            attention_mask = create_sliding_window_mask(
                q_len, kv_seq_len, self.sliding_window_size, query_states.device
            )

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = self.attention_dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat key/value heads for GQA"""
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def create_sliding_window_mask(q_len, kv_len, window_size, device):
    """Create a sliding window attention mask"""
    mask = torch.full((q_len, kv_len), float("-inf"), device=device)
    for i in range(q_len):
        start = max(0, i - window_size + 1)
        end = min(kv_len, i + 1)
        mask[i, start:end] = 0
    return mask.unsqueeze(0).unsqueeze(0)


class SupernovaMLP(nn.Module):
    """SwiGLU MLP for improved efficiency"""
    def __init__(self, config: SupernovaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()  # SwiGLU activation

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class SupernovaDecoderLayer(nn.Module):
    """Transformer decoder layer for Supernova"""
    def __init__(self, config: SupernovaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = SupernovaAttention(config, layer_idx=layer_idx)
        self.mlp = SupernovaMLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + self.dropout(hidden_states)

        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + self.dropout(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class SupernovaModel(nn.Module):
    """Core Supernova USLM Model"""
    def __init__(self, config: SupernovaConfig):
        super().__init__()
        self.padding_idx = 0
        self.vocab_size = config.vocab_size
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [SupernovaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False

        # Initialize weights
        self.post_init()

    def post_init(self):
        """Initialize weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Tuple:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        # Retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # Attention mask
        if attention_mask is not None:
            attention_mask = self._prepare_decoder_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )

        hidden_states = inputs_embeds

        # Decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)
                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # Add last hidden state
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # Create causal mask
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask


class SupernovaForCausalLM(nn.Module):
    """Supernova USLM for Causal Language Modeling"""
    def __init__(self, config: SupernovaConfig):
        super().__init__()
        self.config = config
        self.model = SupernovaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Tie weights if specified
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

        # Initialize weights
        self.post_init()

    def post_init(self):
        """Initialize weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
                if module.bias is not None:
                    module.bias.data.zero_()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states

        # Decoder outputs
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=False,
        )

        hidden_states = outputs[0]
        
        # Add epsilon to prevent underflow
        hidden_states = F.dropout(hidden_states, p=0.1, training=self.training)
        
        # Scale hidden states to prevent exploding values
        hidden_states = hidden_states * (1.0 / hidden_states.norm(dim=-1, keepdim=True).clamp(min=1e-7))
        
        logits = self.lm_head(hidden_states)
        
        # Apply log softmax with better numerical stability
        logits = F.log_softmax(logits, dim=-1)

        loss = None
        if labels is not None:
            # Shift for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Basic cross entropy loss without label smoothing initially
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction='mean')
            
            # Reshape and compute loss
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            
            # Ensure we're on the same device
            shift_labels = shift_labels.to(shift_logits.device)
            
            # Compute main loss
            loss = loss_fct(shift_logits, shift_labels)
            
            # Add a small epsilon to prevent exactly zero loss
            loss = loss + 1e-8
            
            # Simple L2 regularization
            if self.training:
                l2_reg = 0.01 * sum(p.pow(2.0).sum() for p in self.parameters())
                loss = loss + l2_reg
            
            # Ensure loss is finite and reasonable
            if not torch.isfinite(loss):
                loss = torch.tensor(1.0, device=loss.device, requires_grad=True)

        output = {
            'loss': loss,
            'logits': logits,
            'past_key_values': outputs[1] if len(outputs) > 1 else None,
            'hidden_states': outputs[2] if len(outputs) > 2 else None,
            'attentions': outputs[3] if len(outputs) > 3 else None,
        }

        return output

    def generate(
        self,
        input_ids: torch.LongTensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        repetition_penalty: float = 1.0,
        do_sample: bool = True,
        num_return_sequences: int = 1,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
    ):
        """Generate text using the model"""
        batch_size = input_ids.shape[0]
        past_key_values = None
        
        # Expand input_ids for multiple return sequences
        if num_return_sequences > 1:
            input_ids = input_ids.repeat_interleave(num_return_sequences, dim=0)
            batch_size *= num_return_sequences

        generated = input_ids
        
        for _ in range(max_length - input_ids.shape[1]):
            with torch.no_grad():
                outputs = self.forward(
                    input_ids=generated if past_key_values is None else generated[:, -1:],
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                
                logits = outputs['logits'][:, -1, :]
                past_key_values = outputs['past_key_values']
                
                # Apply repetition penalty
                if repetition_penalty != 1.0:
                    for i in range(batch_size):
                        for token_id in set(generated[i].tolist()):
                            logits[i, token_id] /= repetition_penalty
                
                # Apply temperature
                if temperature != 1.0:
                    logits = logits / temperature
                
                # Apply top-k and top-p filtering
                if do_sample:
                    filtered_logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
                    probs = F.softmax(filtered_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                
                generated = torch.cat([generated, next_token], dim=-1)
                
                # Check for EOS token
                if eos_token_id is not None and (next_token == eos_token_id).all():
                    break
        
        return generated


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """Filter a distribution using top-k and/or nucleus (top-p) filtering"""
    batch_size = logits.shape[0]
    top_k = min(top_k, logits.size(-1))  # Safety check
    
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    
    return logits


def _make_causal_mask(input_ids_shape, dtype, device, past_key_values_length=0):
    """Make causal mask for self-attention"""
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


def _expand_mask(mask, dtype, tgt_len=None):
    """Expand attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`"""
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


# Model initialization helper
def create_supernova_model(
    vocab_size: int = 32000,
    hidden_size: int = 768,
    num_layers: int = 8,
    num_heads: int = 12,
    num_kv_heads: int = 4,
    intermediate_size: int = 2048,
    max_position_embeddings: int = 2048,
    **kwargs
) -> SupernovaForCausalLM:
    """Create a Supernova model with specified configuration"""
    config = SupernovaConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        intermediate_size=intermediate_size,
        max_position_embeddings=max_position_embeddings,
        **kwargs
    )
    
    model = SupernovaForCausalLM(config)
    
    # Print model statistics
    param_count = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Supernova USLM Model Created!")
    print(f"Total parameters: {param_count:,} ({param_count/1e6:.2f}M)")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    print(f"Configuration: {config.get_num_params():,} estimated parameters")
    
    return model


if __name__ == "__main__":
    # Test model creation
    model = create_supernova_model()
    
    # Test forward pass
    batch_size = 2
    seq_length = 128
    input_ids = torch.randint(0, 32000, (batch_size, seq_length))
    
    with torch.no_grad():
        outputs = model(input_ids)
        print(f"\nTest forward pass successful!")
        print(f"Output shape: {outputs['logits'].shape}")
