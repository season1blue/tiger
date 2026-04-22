import numpy as np
import math
import warnings
from typing import List, Optional, Tuple, Union
import safetensors
import importlib
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

import transformers

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from transformers.generation.logits_process import LogitsProcessorList

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from llava.mm_utils import get_anyres_image_grid_shape
from llava.model.llava_arch import unpad_image

import llava.model.llava_arch
logger = logging.get_logger(__name__)


def _fit_tensor_to_shape(source, target_shape):
    if source is None:
        return None
    if not isinstance(source, torch.Tensor):
        source = torch.as_tensor(source)

    target_numel = 1
    for dim in target_shape:
        target_numel *= dim

    flat_source = source.reshape(-1)
    if flat_source.numel() == 0:
        return torch.zeros(target_shape, dtype=source.dtype, device=source.device)

    repeat_count = math.ceil(target_numel / flat_source.numel())
    resized = flat_source.repeat(repeat_count)[:target_numel]
    return resized.reshape(target_shape).to(device=source.device, dtype=source.dtype)


def _legacy_cache_seq_length(past_key_values) -> int:
    if past_key_values is None:
        return 0
    try:
        first_layer = past_key_values[0]
        if isinstance(first_layer, (tuple, list)) and len(first_layer) > 0:
            first_k = first_layer[0]
            if isinstance(first_k, torch.Tensor) and first_k.ndim >= 3:
                return int(first_k.shape[-2])
    except Exception:
        pass
    return 0

class LlamaMLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

        # MemVR
        self.apply_memvr = False
        self.vision_retracing_layer = 0
        self.visual_token = None
        self.retracing_ratio = 0
        self.entropy_threshold = 1
        self.starting_layer = 0
        self.ending_layer = 0
        self.retrace_delay_layers = 1
        self.retrace_target_layers = ""
        self.memvr_method = "memvr"
        self.state_drift_threshold = 0.5
        self.state_drift_pooling = "mean"
        self.image_token_mask = None
        self.adpt_sign = 0




    def forward(self, x):
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        elif self.adpt_sign == 0:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        elif self.adpt_sign == 1:
            ffn_out = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
            adapter_out = torch.matmul(torch.matmul(x, self.adpt_w1.T), self.adpt_w2.T)
            norm_adapter_out = (torch.mean(torch.abs(ffn_out)) / torch.mean(torch.abs((adapter_out)))) * adapter_out
            return (ffn_out*(1-self.retracing_ratio) + norm_adapter_out*self.retracing_ratio)
            

        return down_proj



def forward(
    self,
    input_ids: torch.LongTensor, #= None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    logits_processor = LogitsProcessorList() ,
) -> Union[Tuple, BaseModelOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError(
            "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
        )

    if self.gradient_checkpointing and self.training and use_cache:
        logger.warning_once(
            "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
        )
        use_cache = False

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    past_seen_tokens = 0
    if use_cache:  # kept for BC (cache positions)
        if isinstance(past_key_values, StaticCache):
            past_seen_tokens = past_key_values.get_seq_length()
        elif isinstance(past_key_values, Cache):
            past_seen_tokens = past_key_values.get_seq_length()
        elif past_key_values is not None and hasattr(DynamicCache, "from_legacy_cache"):
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_seen_tokens = past_key_values.get_seq_length()
        else:
            # Fallback for versions where DynamicCache legacy helpers are removed.
            past_seen_tokens = _legacy_cache_seq_length(past_key_values)

    if cache_position is None:
        if isinstance(past_key_values, StaticCache):
            raise ValueError("cache_position is a required argument when using StaticCache.")
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )

    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    causal_mask = self._update_causal_mask(attention_mask, inputs_embeds, cache_position, past_seen_tokens)

    # embed positions
    hidden_states = inputs_embeds

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None

    # MemVR
    layer = 0
    entropy_list = []
    apply_memvr = self.layers[0].mlp.apply_memvr
    visual_token = self.layers[0].mlp.visual_token
    dynamic_visual_token = visual_token
    retracing_ratio = self.layers[0].mlp.retracing_ratio
    entropy_threshold = self.layers[0].mlp.entropy_threshold
    starting_layer = self.layers[0].mlp.starting_layer
    ending_layer = self.layers[0].mlp.ending_layer
    method = str(getattr(self.layers[0].mlp, "memvr_method", "memvr") or "memvr").lower()
    retrace_delay_layers = max(1, int(getattr(self.layers[0].mlp, "retrace_delay_layers", 1)))
    state_drift_threshold = float(getattr(self.layers[0].mlp, "state_drift_threshold", 0.5))
    state_drift_pooling = str(getattr(self.layers[0].mlp, "state_drift_pooling", "mean") or "mean").lower()
    image_token_mask = getattr(self.layers[0].mlp, "image_token_mask", None) if past_seen_tokens == 0 else None
    visual_retracing_event = False # to prevent multiple retracing event
    vision_retracing_sign  = False # to decide whether to add visual token in the next layer
    prev_prev_img_state = None
    prev_img_state = None
    state_drift_score = 0.0
    state_drift_ready = False


    for decoder_layer in self.layers:
        # print("/n calculating hidden states at layer: ", layer)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states,
                causal_mask,
                position_ids,
                past_key_values,
                output_attentions,
                use_cache,
                cache_position,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
            )

        hidden_states = layer_outputs[0]

        if method in {"memvr", "evo"} and image_token_mask is not None:
            dynamic_visual_token = hidden_states[image_token_mask]

        if method == "evo" and image_token_mask is not None and hidden_states.dim() == 3:
            img_mask = image_token_mask
            if img_mask.dim() > 2:
                img_mask = img_mask.squeeze(-1)
            if img_mask.dim() == 2 and img_mask.shape[1] == hidden_states.shape[1]:
                img_mask = img_mask.to(device=hidden_states.device, dtype=torch.bool)
                valid_img_samples = img_mask.any(dim=1)
                if valid_img_samples.any():
                    img_mask_f = img_mask.unsqueeze(-1).to(dtype=hidden_states.dtype)
                    pooled_img_state = (hidden_states * img_mask_f).sum(dim=1) / img_mask_f.sum(dim=1).clamp_min(1.0)

                    state_drift_ready = False
                    if prev_img_state is not None and prev_prev_img_state is not None:
                        v_prev = prev_img_state - prev_prev_img_state
                        v_curr = pooled_img_state - prev_img_state
                        cos_sim = F.cosine_similarity(v_curr, v_prev, dim=-1, eps=1e-6)
                        drift_per_sample = 1.0 - cos_sim
                        drift_per_sample = torch.where(
                            valid_img_samples,
                            drift_per_sample,
                            torch.zeros_like(drift_per_sample),
                        )

                        if state_drift_pooling == "max":
                            state_drift_score = float(drift_per_sample.max().item())
                        else:
                            state_drift_score = float(drift_per_sample[valid_img_samples].mean().item())
                        state_drift_ready = True

                    prev_prev_img_state = prev_img_state
                    prev_img_state = pooled_img_state

        if use_cache:
            next_decoder_cache = layer_outputs[2 if output_attentions else 1]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

        # print("\n calculating logits at layer: ", layer)

        norm_hidden_states = self.norm(hidden_states)
        logits = self.lm_head(norm_hidden_states)
        logits = logits[:, -1, :]
        logits = logits.float()
        logits = logits_processor(input_ids, logits)

        # Calculate the layer entropy
        top_k_scores, top_k_indices = torch.topk(logits, 10)
        probabilities = F.softmax(top_k_scores, dim=-1)
        entropy = torch.sum((-probabilities[:10] * torch.log(probabilities[:10]))/np.log(10))
        entropy = entropy.item()

        # formatted_top_k_scores = [f"{score:.3f}" for score in top_k_scores.flatten().tolist()]
        # formatted_top_k_indices = [f"{index:.3f}" for index in top_k_indices.flatten().tolist()]
        # formatted_probabilities = [f"{prob:.3f}" for prob in probabilities.flatten().tolist()]
        formatted_entropy = f"{entropy:.3f}"


        # round n+1
        # vision_retracing_sign is true, meaning that the visual token has been added. Now, clear the adaptation channel, reset the adpt_sign and vision_retracing_sign.
        if vision_retracing_sign == True:

            self.layers[layer].mlp.adpt_sign = 0
            self.layers[layer].mlp.adpt_w1 = torch.nn.Parameter(torch.zeros_like(visual_token))
            self.layers[layer].mlp.adpt_w2 = torch.nn.Parameter(torch.zeros_like(visual_token.T))
            # print("\n added visual token with adatption channel at layer ", layer)
                
            vision_retracing_sign = False


            
        # round n
        # calculate the entropy of the top 10 logits. if the entropy is greater than the threshold, and the visual retracing event is not happening, and the layer is within the range of starting and ending layer, then add the visual token to the next layer with adaptation channel
        # initialize the adaptation channel with the visual token
        trigger_hit = False
        if method == "evo":
            trigger_hit = state_drift_ready and (state_drift_score > state_drift_threshold)
        else:
            trigger_hit = entropy > entropy_threshold

        if trigger_hit and visual_retracing_event == False and layer > starting_layer and layer < ending_layer and layer + retrace_delay_layers < len(self.layers):
            
            vision_retracing_sign = True
            visual_retracing_event = True

            target_layer = layer + retrace_delay_layers
            self.layers[target_layer].mlp.adpt_sign = 1 # triggers the MemVR adaptation channel in MLP of the next layer

            adapter_seed = dynamic_visual_token if dynamic_visual_token is not None else visual_token
            if adapter_seed is not None:
                if isinstance(adapter_seed, torch.Tensor) and adapter_seed.dim() > 2:
                    adapter_seed = adapter_seed[0]
                if isinstance(adapter_seed, torch.Tensor) and adapter_seed.dim() == 1:
                    adapter_seed = adapter_seed.unsqueeze(0)
                adapter_seed = adapter_seed.to(dtype=hidden_states.dtype, device=hidden_states.device)

                next_mlp = self.layers[target_layer].mlp
                next_mlp.adpt_w1 = torch.nn.Parameter(torch.zeros_like(adapter_seed))
                next_mlp.adpt_w2 = torch.nn.Parameter(torch.zeros_like(adapter_seed.T))
                next_mlp.adpt_w1 += (torch.mean(torch.abs(next_mlp.up_proj.weight)) / (torch.mean(torch.abs(adapter_seed)) + 1e-6)) * adapter_seed
                next_mlp.adpt_w2 += (torch.mean(torch.abs(next_mlp.down_proj.weight)) / (torch.mean(torch.abs(adapter_seed)) + 1e-6)) * adapter_seed.T
                next_mlp.retracing_ratio = retracing_ratio

            
        # print("Extracted Top 10 largest scores:", formatted_top_k_scores)
        # print("Extracted Indices of top 10 largest scores:", formatted_top_k_indices)
        # print("Probabilities of top 10 logits:", formatted_probabilities)
        # print("Entropy of top 10 logits:", formatted_entropy, "\n")
        entropy_list.append(formatted_entropy)
        layer += 1

    hidden_states = self.norm(hidden_states)
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = None
    if use_cache:
        next_cache = (
            next_decoder_cache.to_legacy_cache()
            if isinstance(next_decoder_cache, Cache) and hasattr(next_decoder_cache, "to_legacy_cache")
            else next_decoder_cache
        )
    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )

def prepare_inputs_labels_for_multimodal(
    self, input_ids, position_ids, attention_mask, past_key_values, labels,
    images, image_sizes=None
):
    vision_tower = self.get_vision_tower()
    if vision_tower is None or images is None or input_ids.shape[1] == 1:
        return input_ids, position_ids, attention_mask, past_key_values, None, labels

    if type(images) is list or images.ndim == 5:
        if type(images) is list:
            images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
        concat_images = torch.cat([image for image in images], dim=0)
        image_features = self.encode_images(concat_images)
        split_sizes = [image.shape[0] for image in images]
        image_features = torch.split(image_features, split_sizes, dim=0)
        mm_patch_merge_type = getattr(self.config, 'mm_patch_merge_type', 'flat')
        image_aspect_ratio = getattr(self.config, 'image_aspect_ratio', 'square')
        if mm_patch_merge_type == 'flat':
            image_features = [x.flatten(0, 1) for x in image_features]
        elif mm_patch_merge_type.startswith('spatial'):
            new_image_features = []
            for image_idx, image_feature in enumerate(image_features):
                if image_feature.shape[0] > 1:
                    base_image_feature = image_feature[0]
                    image_feature = image_feature[1:]
                    height = width = self.get_vision_tower().num_patches_per_side
                    assert height * width == base_image_feature.shape[0]
                    if image_aspect_ratio == 'anyres':
                        num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, self.get_vision_tower().config.image_size)
                        image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                    else:
                        raise NotImplementedError
                    if 'unpad' in mm_patch_merge_type:
                        image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                        image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                        image_feature = unpad_image(image_feature, image_sizes[image_idx])
                        image_feature = torch.cat((
                            image_feature,
                            self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)
                        ), dim=-1)
                        image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                    else:
                        image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                        image_feature = image_feature.flatten(0, 3)
                    image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                else:
                    image_feature = image_feature[0]
                    if 'unpad' in mm_patch_merge_type:
                        image_feature = torch.cat((
                            image_feature,
                            self.model.image_newline[None].to(image_feature.device)
                        ), dim=0)
                new_image_features.append(image_feature)
            image_features = new_image_features
        else:
            raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
    else:
        image_features = self.encode_images(images)

    # TODO: image start / end is not implemented here to support pretraining.
    if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
        raise NotImplementedError

    # Let's just add dummy tensors if they do not exist,
    # it is a headache to deal with None all the time.
    # But it is not ideal, and if you have a better idea,
    # please open an issue / submit a PR, thanks.
    _labels = labels
    _position_ids = position_ids
    _attention_mask = attention_mask
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    else:
        attention_mask = attention_mask.bool()
    if position_ids is None:
        position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
    if labels is None:
        labels = torch.full_like(input_ids, IGNORE_INDEX)

    # remove the padding using attention_mask -- FIXME
    _input_ids = input_ids
    input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
    labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

    new_input_embeds = []
    new_labels = []
    cur_image_idx = 0
    for batch_idx, cur_input_ids in enumerate(input_ids):
        num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
        if num_images == 0:
            cur_image_features = image_features[cur_image_idx]
            cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
            cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
            new_input_embeds.append(cur_input_embeds)
            new_labels.append(labels[batch_idx])
            cur_image_idx += 1
            continue

        image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
        cur_input_ids_noim = []
        cur_labels = labels[batch_idx]
        cur_labels_noim = []
        for i in range(len(image_token_indices) - 1):
            cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
            cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
        split_sizes = [x.shape[0] for x in cur_labels_noim]
        cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
        cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
        cur_new_input_embeds = []
        cur_new_labels = []

        for i in range(num_images + 1):
            cur_new_input_embeds.append(cur_input_embeds_no_im[i])
            cur_new_labels.append(cur_labels_noim[i])
            if i < num_images:
                cur_image_features = image_features[cur_image_idx]
                cur_image_idx += 1
                cur_new_input_embeds.append(cur_image_features)
                cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

        cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

        cur_new_input_embeds = torch.cat(cur_new_input_embeds)
        cur_new_labels = torch.cat(cur_new_labels)

        new_input_embeds.append(cur_new_input_embeds)
        new_labels.append(cur_new_labels)

    # Truncate sequences to max length as image embeddings can make the sequence longer
    tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
    if tokenizer_model_max_length is not None:
        new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
        new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

    # Combine them
    max_len = max(x.shape[0] for x in new_input_embeds)
    batch_size = len(new_input_embeds)

    new_input_embeds_padded = []
    new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
    attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
    position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

    for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
        cur_len = cur_new_embed.shape[0]
        if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
            new_input_embeds_padded.append(torch.cat((
                torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                cur_new_embed
            ), dim=0))
            if cur_len > 0:
                new_labels_padded[i, -cur_len:] = cur_new_labels
                attention_mask[i, -cur_len:] = True
                position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
        else:
            new_input_embeds_padded.append(torch.cat((
                cur_new_embed,
                torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
            ), dim=0))
            if cur_len > 0:
                new_labels_padded[i, :cur_len] = cur_new_labels
                attention_mask[i, :cur_len] = True
                position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

    new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

    if _labels is None:
        new_labels = None
    else:
        new_labels = new_labels_padded

    if _attention_mask is None:
        attention_mask = None
    else:
        attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

    if _position_ids is None:
        position_ids = None

    # MemVR
    # pass the image features to the first layer of the model
    self.model.layers[0].mlp.visual_token = cur_image_features
    
    return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels


class QWenMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.w1 = nn.Linear(
            config.hidden_size, config.intermediate_size // 2, bias=not config.no_bias
        )
        self.w2 = nn.Linear(
            config.hidden_size, config.intermediate_size // 2, bias=not config.no_bias
        )
        ff_dim_in = config.intermediate_size // 2
        self.c_proj = nn.Linear(ff_dim_in, config.hidden_size, bias=not config.no_bias)

        # MemVR
        self.apply_memvr = False
        self.vision_token = None
        self.retracing_ratio = 0
        self.entropy_threshold = 1
        self.starting_layer = 0
        self.ending_layer = 0
        self.adpt_sign = 0
        self.adpt_w1 = None
        self.adpt_w2 = None

    def forward(self, hidden_states):
        return qwen_mlp_forward(self, hidden_states)


def qwen_mlp_forward(self, hidden_states):
    a1 = self.w1(hidden_states)
    a2 = self.w2(hidden_states)
    intermediate_parallel = a1 * F.silu(a2)
    output = self.c_proj(intermediate_parallel)

    if getattr(self, "adpt_sign", 0) == 1:
        adpt_w1 = getattr(self, "adpt_w1", None)
        adpt_w2 = getattr(self, "adpt_w2", None)
        if adpt_w1 is not None and adpt_w2 is not None:
            adapter_source = hidden_states[0] if hidden_states.dim() == 3 else hidden_states
            if adapter_source.dim() == 1:
                adapter_source = adapter_source.unsqueeze(0)
            if adpt_w1.dim() == 3:
                adpt_w1 = adpt_w1[0]
            if adpt_w2.dim() == 3:
                adpt_w2 = adpt_w2[0]

            adapter_out = torch.matmul(F.silu(torch.matmul(adapter_source, adpt_w1.T)), adpt_w2.T)
            if hidden_states.dim() == 3 and adapter_out.dim() == 2:
                adapter_out = adapter_out.unsqueeze(0)

            eps = 1e-6
            norm_scale = torch.mean(torch.abs(output)) / (torch.mean(torch.abs(adapter_out)) + eps)
            retracing_ratio = float(getattr(self, "retracing_ratio", 0.0))
            return output * (1 - retracing_ratio) + norm_scale * adapter_out * retracing_ratio

    return output

def qwen_model_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    token_type_ids: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    encoder_attention_mask: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    logits_processor=LogitsProcessorList(),
):
    visual_config = getattr(self.config, "visual", None)
    image_start_id = None
    if isinstance(visual_config, dict):
        image_start_id = visual_config.get("image_start_id")
    elif visual_config is not None:
        image_start_id = getattr(visual_config, "image_start_id", None)

    if (
        past_key_values is None
        and input_ids is not None
        and image_start_id is not None
        and torch.any(input_ids == image_start_id)
    ):
        bos_pos = torch.where(input_ids == image_start_id)
        eos_pos = torch.where(input_ids == image_start_id + 1)
        assert (bos_pos[0] == eos_pos[0]).all()
        img_pos = torch.stack((bos_pos[0], bos_pos[1], eos_pos[1]), dim=1)
        image_paths = []
        for i, a, b in img_pos:
            image = input_ids[i][a + 1 : b - 1].tolist()
            image = image[: image.index(image_start_id + 2)]
            image_paths.append(bytes(image).decode("utf-8"))

        images = self.visual.encode(image_paths)
        assert images.shape[0] == len(image_paths)
        fake_images = None
    elif self.training:
        fake_images = torch.zeros(1, 3, 224, 224).to(
            dtype=self.visual.conv1.weight.dtype,
            device=self.visual.conv1.weight.device,
        )
        images = self.visual(fake_images)
        img_pos = []
    else:
        fake_images = None
        images = None
        img_pos = []

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
    elif input_ids is not None:
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        batch_size = input_ids.shape[0]
    elif inputs_embeds is not None:
        input_shape = inputs_embeds.size()[:-1]
        batch_size = inputs_embeds.shape[0]
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")

    device = input_ids.device if input_ids is not None else inputs_embeds.device

    if token_type_ids is not None:
        token_type_ids = token_type_ids.view(-1, input_shape[-1])
    if position_ids is not None:
        position_ids = position_ids.view(-1, input_shape[-1])

    if past_key_values is None:
        past_length = 0
        past_key_values = tuple([None] * len(self.h))
    else:
        past_length = past_key_values[0][0].size(-2)

    if position_ids is None:
        position_ids = torch.arange(
            past_length,
            input_shape[-1] + past_length,
            dtype=torch.long,
            device=device,
        )
        position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

    encoder_attention_mask = None
    head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

    if inputs_embeds is None:
        inputs_embeds = self.wte(input_ids)

    if batch_size <= 0:
        raise ValueError("batch_size has to be defined and > 0")
    attention_mask = self._prepare_decoder_attention_mask(
        attention_mask, input_shape, inputs_embeds, past_length
    )

    hidden_states = inputs_embeds

    kv_seq_len = hidden_states.size()[1]
    if past_key_values[0] is not None:
        kv_seq_len += past_key_values[0][0].shape[1]
    if self.use_dynamic_ntk and kv_seq_len == hidden_states.size()[1] and not self.training:
        context_value = math.log(kv_seq_len / self.seq_length, 2) + 1
        ntk_alpha = 2 ** math.ceil(context_value) - 1
        ntk_alpha = max(ntk_alpha, 1)
    else:
        ntk_alpha = self.rotary_emb._ntk_alpha_cached

    rotary_pos_emb = self.rotary_emb(kv_seq_len, ntk_alpha=ntk_alpha)
    for idx in range(len(rotary_pos_emb)):
        rotary_pos_emb[idx] = rotary_pos_emb[idx].to(hidden_states.device)

    drop_layer = getattr(self, "drop", None)
    if callable(drop_layer):
        hidden_states = drop_layer(hidden_states).clone()
    else:
        hidden_states = hidden_states.clone()

    if fake_images is not None:
        hidden_states = hidden_states + images.mean() * 0
    elif images is not None:
        for idx, (i, a, b) in enumerate(img_pos):
            hidden_states[i][a + 1 : b] = images[idx]

    if isinstance(images, torch.Tensor):
        self.h[0].mlp.vision_token = images[0] if images.dim() > 2 else images
    else:
        self.h[0].mlp.vision_token = None

    if self.gradient_checkpointing and self.training and use_cache:
        logger.warning_once("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.")
        use_cache = False

    next_decoder_cache = () if use_cache else None
    all_self_attns = () if output_attentions else None
    all_hidden_states = () if output_hidden_states else None

    # MemVR control flow: same idea as llama forward
    layer = 0
    entropy_list = []
    apply_memvr = getattr(self.h[0].mlp, "apply_memvr", False)
    visual_token = getattr(self.h[0].mlp, "vision_token", None)
    retracing_ratio = getattr(self.h[0].mlp, "retracing_ratio", 0.0)
    entropy_threshold = getattr(self.h[0].mlp, "entropy_threshold", 1.0)
    starting_layer = getattr(self.h[0].mlp, "starting_layer", 0)
    ending_layer = getattr(self.h[0].mlp, "ending_layer", len(self.h) - 1)
    visual_retracing_event = False
    vision_retracing_sign = False

    for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # round n+1
        # vision_retracing_sign is true, meaning that the visual token has been added.
        # Now, clear the adaptation channel, reset adpt_sign and vision_retracing_sign.
        if vision_retracing_sign == True:
            self.h[layer].mlp.adpt_sign = 0
            if visual_token is not None:
                adapter_seed = visual_token
                if isinstance(adapter_seed, torch.Tensor) and adapter_seed.dim() > 2:
                    adapter_seed = adapter_seed[0]
                if isinstance(adapter_seed, torch.Tensor) and adapter_seed.dim() == 1:
                    adapter_seed = adapter_seed.unsqueeze(0)
                adapter_seed = adapter_seed.to(dtype=hidden_states.dtype, device=hidden_states.device)

                cur_mlp = self.h[layer].mlp
                cur_w1_seed = _fit_tensor_to_shape(adapter_seed, cur_mlp.w1.weight.shape)
                cur_w2_seed = _fit_tensor_to_shape(adapter_seed.T, cur_mlp.c_proj.weight.shape)
                cur_mlp.adpt_w1 = torch.nn.Parameter(torch.zeros_like(cur_w1_seed))
                cur_mlp.adpt_w2 = torch.nn.Parameter(torch.zeros_like(cur_w2_seed))
            else:
                self.h[layer].mlp.adpt_w1 = None
                self.h[layer].mlp.adpt_w2 = None

            vision_retracing_sign = False

        if self.gradient_checkpointing and self.training:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs, use_cache, output_attentions)

                return custom_forward

            outputs = torch.utils.checkpoint.checkpoint(
                create_custom_forward(block),
                hidden_states,
                rotary_pos_emb,
                self.registered_causal_mask,
                None,
                attention_mask,
                head_mask[i],
                encoder_hidden_states,
                encoder_attention_mask,
            )
        else:
            outputs = block(
                hidden_states,
                layer_past=layer_past,
                rotary_pos_emb=rotary_pos_emb,
                registered_causal_mask=self.registered_causal_mask,
                attention_mask=attention_mask,
                head_mask=head_mask[i],
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )

        hidden_states = outputs[0]

        if use_cache:
            layer_cache = outputs[2 if output_attentions else 1]
            next_decoder_cache = next_decoder_cache + (layer_cache,)

        if output_attentions:
            all_self_attns = all_self_attns + (outputs[1],)

        # calculate logits/entropy at each layer (aligned with llama-style flow)
        norm_hidden_states = self.ln_f(hidden_states)
        logits = self.lm_head(norm_hidden_states)
        logits = logits[:, -1, :]
        logits = logits.float()
        logits = logits_processor(input_ids, logits)

        top_k = min(10, logits.shape[-1])
        top_k_scores, top_k_indices = torch.topk(logits, top_k)
        probabilities = F.softmax(top_k_scores, dim=-1)
        entropy_base = np.log(max(top_k, 2))
        entropy = torch.sum((-probabilities * torch.log(probabilities + 1e-12)) / entropy_base)
        entropy_value = float(entropy.item())
        formatted_entropy = f"{entropy_value:.3f}"
        entropy_list.append(formatted_entropy)

        # round n
        # calculate entropy of top-k logits; when above threshold and in layer range,
        # add visual token to the next layer adaptation channel.
        if (
            apply_memvr
            and
            entropy_value > entropy_threshold
            and not visual_retracing_event
            and layer > starting_layer
            and layer < ending_layer
            and layer + 1 < len(self.h)
        ):
            vision_retracing_sign = True
            visual_retracing_event = True

            next_mlp = self.h[layer + 1].mlp
            next_mlp.adpt_sign = 1

            if visual_token is not None:
                adapter_seed = visual_token
                if isinstance(adapter_seed, torch.Tensor) and adapter_seed.dim() > 2:
                    adapter_seed = adapter_seed[0]
                if isinstance(adapter_seed, torch.Tensor) and adapter_seed.dim() == 1:
                    adapter_seed = adapter_seed.unsqueeze(0)
                adapter_seed = adapter_seed.to(dtype=hidden_states.dtype, device=hidden_states.device)

                next_w1_seed = _fit_tensor_to_shape(adapter_seed, next_mlp.w1.weight.shape)
                next_w2_seed = _fit_tensor_to_shape(adapter_seed.T, next_mlp.c_proj.weight.shape)
                next_mlp.adpt_w1 = torch.nn.Parameter(torch.zeros_like(next_w1_seed))
                next_mlp.adpt_w2 = torch.nn.Parameter(torch.zeros_like(next_w2_seed))

                scale_w1 = torch.mean(torch.abs(next_mlp.w1.weight)) / (torch.mean(torch.abs(next_w1_seed)) + 1e-6)
                scale_w2 = torch.mean(torch.abs(next_mlp.c_proj.weight)) / (torch.mean(torch.abs(next_w2_seed)) + 1e-6)
                next_mlp.adpt_w1 += scale_w1 * next_w1_seed
                next_mlp.adpt_w2 += scale_w2 * next_w2_seed
                next_mlp.retracing_ratio = retracing_ratio

        layer += 1

    hidden_states = self.ln_f(hidden_states)
    hidden_states = hidden_states.view(input_shape + (hidden_states.size(-1),))
    if output_hidden_states:
        all_hidden_states = all_hidden_states + (hidden_states,)

    if not return_dict:
        return tuple(v for v in [hidden_states, next_decoder_cache, all_hidden_states, all_self_attns] if v is not None)

    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_decoder_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )
    
import ipdb
def apply_memvr_llama(
        self,
        starting_layer: int,
        ending_layer: int,
        entropy_threshold: float,
        retracing_ratio: float,
        method: str = "memvr",
        retrace_delay_layers: int = 1,
        state_drift_threshold: float = 0.5,
        state_drift_pooling: str = "mean",
    ):
    transformers.models.llama.modeling_llama.LlamaMLP = LlamaMLP
    transformers.models.llama.modeling_llama.LlamaModel.forward = forward
    llava.model.llava_arch.LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal = prepare_inputs_labels_for_multimodal

    self.model.lm_head = self.lm_head
    self.model.layers[0].mlp.apply_memvr = True
    self.model.layers[0].mlp.starting_layer = starting_layer
    self.model.layers[0].mlp.ending_layer = ending_layer
    self.model.layers[0].mlp.entropy_threshold = entropy_threshold
    self.model.layers[0].mlp.retrace_delay_layers = max(1, int(retrace_delay_layers))
    self.model.layers[0].mlp.memvr_method = str(method)
    self.model.layers[0].mlp.state_drift_threshold = float(state_drift_threshold)
    self.model.layers[0].mlp.state_drift_pooling = str(state_drift_pooling)
    for layer in range(31):
        self.model.layers[layer].mlp.retracing_ratio = retracing_ratio
        self.model.layers[layer].mlp.image_token_mask = None

def apply_memvr_qwen(
        self,
        starting_layer: int,
        ending_layer: int,
        entropy_threshold: float,
        retracing_ratio: float
    ):
    qwen_module = importlib.import_module(self.transformer.__class__.__module__)

    qwen_module.QWenMLP = QWenMLP
    if hasattr(qwen_module, "QWenModel"):
        qwen_module.QWenModel.forward = qwen_model_forward
    
    
    # Also patch currently-instantiated runtime classes so existing model objects take effect immediately.
    type(self.transformer).forward = qwen_model_forward
    for block in self.transformer.h:
        type(block.mlp).forward = qwen_mlp_forward

    qwen_mlp_cls = getattr(qwen_module, "QWenMLP", type(self.transformer.h[0].mlp))
    qwen_model_cls = getattr(qwen_module, "QWenModel", type(self.transformer))
    qwen_mlp_cls.forward = qwen_mlp_forward
    qwen_model_cls.forward = qwen_model_forward

    print("QWenMLP symbol replaced:", qwen_module.QWenMLP is QWenMLP)
    print("QWenModel.forward replaced:", type(self.transformer).forward is qwen_model_forward)
    print("MLP instance class forward replaced:", type(self.transformer.h[0].mlp).forward is qwen_mlp_forward)

    self.transformer.lm_head = self.lm_head

    num_layers = len(self.transformer.h)
    for layer in range(num_layers):
        mlp = self.transformer.h[layer].mlp
        mlp.apply_memvr = True
        mlp.starting_layer = starting_layer
        mlp.ending_layer = ending_layer
        mlp.entropy_threshold = entropy_threshold
        mlp.retracing_ratio = retracing_ratio
        mlp.vision_retracing_method = "adapt"
        mlp.vision_retracing_sign = False
        mlp.vision_retracing_event = False
        mlp.vision_token = None
        mlp.adpt_sign = 0
        mlp.adpt_w1 = None
        mlp.adpt_w2 = None


def apply_memvr_qwen25(
        self,
        starting_layer: int,
        ending_layer: int,
        entropy_threshold: float,
    retracing_ratio: float,
    retrace_delay_layers: int = 1,
    retrace_target_layers: str = "",
    method: str = "memvr",
    state_drift_threshold: float = 0.5,
    state_drift_pooling: str = "mean",
    ):
    # Qwen2.5-VL MemVR logic is directly implemented in modeling_qwen2_5_vl.py.
    self.model.language_model.lm_head = self.lm_head

    self.model.language_model.layers[0].mlp.apply_memvr = True
    self.model.language_model.layers[0].mlp.starting_layer = starting_layer
    self.model.language_model.layers[0].mlp.ending_layer = ending_layer
    self.model.language_model.layers[0].mlp.entropy_threshold = entropy_threshold
    self.model.language_model.layers[0].mlp.retrace_delay_layers = max(1, int(retrace_delay_layers))
    self.model.language_model.layers[0].mlp.retrace_target_layers = retrace_target_layers
    self.model.language_model.layers[0].mlp.memvr_method = str(method)
    self.model.language_model.layers[0].mlp.state_drift_threshold = float(state_drift_threshold)
    self.model.language_model.layers[0].mlp.state_drift_pooling = str(state_drift_pooling)
    for layer in range(28):
        self.model.language_model.layers[layer].mlp.retracing_ratio = retracing_ratio
        

    # num_layers = len(self.model.language_model.layers)
    # for layer in range(num_layers):
    #     mlp = self.model.language_model.layers[layer].mlp
    #     mlp.apply_memvr = True
    #     mlp.starting_layer = starting_layer
    #     mlp.ending_layer = ending_layer
    #     mlp.entropy_threshold = entropy_threshold
    #     mlp.retracing_ratio = retracing_ratio
    #     mlp.vision_retracing_method = "adapt"
    #     mlp.vision_retracing_sign = False
    #     mlp.vision_retracing_event = False
    #     mlp.vision_token = None
    #     mlp.adpt_sign = 0
    #     mlp.adpt_w1 = None
    #     mlp.adpt_w2 = None


def apply_memvr_qwen35(
        self,
        starting_layer: int,
        ending_layer: int,
        entropy_threshold: float,
        retracing_ratio: float,
        retrace_delay_layers: int = 1,
        retrace_target_layers: str = "",
        method: str = "memvr",
        state_drift_threshold: float = 0.5,
        state_drift_pooling: str = "mean",
    ):
    # Qwen3.5-VL MemVR logic is directly implemented in modeling_qwen3_5.py.
    self.model.language_model.lm_head = self.lm_head

    self.model.language_model.layers[0].mlp.apply_memvr = True
    self.model.language_model.layers[0].mlp.starting_layer = starting_layer
    self.model.language_model.layers[0].mlp.ending_layer = ending_layer
    self.model.language_model.layers[0].mlp.entropy_threshold = entropy_threshold
    self.model.language_model.layers[0].mlp.retrace_delay_layers = max(1, int(retrace_delay_layers))
    self.model.language_model.layers[0].mlp.retrace_target_layers = retrace_target_layers
    self.model.language_model.layers[0].mlp.memvr_method = str(method)
    self.model.language_model.layers[0].mlp.state_drift_threshold = float(state_drift_threshold)
    self.model.language_model.layers[0].mlp.state_drift_pooling = str(state_drift_pooling)

    num_layers = len(self.model.language_model.layers)
    for layer in range(num_layers):
        mlp = self.model.language_model.layers[layer].mlp
        mlp.retracing_ratio = retracing_ratio
        if not hasattr(mlp, "apply_memvr"):
            mlp.apply_memvr = False
        if not hasattr(mlp, "visual_token"):
            mlp.visual_token = None
        if not hasattr(mlp, "adpt_sign"):
            mlp.adpt_sign = 0
        if not hasattr(mlp, "adpt_w1"):
            mlp.adpt_w1 = None
        if not hasattr(mlp, "adpt_w2"):
            mlp.adpt_w2 = None