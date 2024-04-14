from typing import Optional, Tuple, Union
import torch
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.clip.configuration_clip import CLIPConfig
from transformers.models.clip.modeling_clip import CLIPModel, CLIPTextTransformer, _make_causal_mask, _expand_mask, clip_loss, CLIPOutput


class CLIPTextTransformerCanReceiveEmbed(CLIPTextTransformer):
    def forward(self,
        input_ids: Optional[torch.Tensor] = None,
        input_embeds: Optional[torch.Tensor] = None, # NOTE
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,) -> Union[Tuple, BaseModelOutputWithPooling]:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        if input_embeds is None:
            if input_ids is None:
                raise ValueError("You have to specify input_ids")

            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])

            hidden_states = self.embeddings(input_ids=input_ids, position_ids=position_ids)
        else:
            hidden_states = input_embeds
            input_shape = torch.Size([hidden_states.size(0), hidden_states.size(1)])

        # CLIP's text model uses causal mask, prepare it here.
        # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
        # print(input_shape)
        causal_attention_mask = _make_causal_mask(input_shape, hidden_states.dtype, device=hidden_states.device)
        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, hidden_states.dtype)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.final_layer_norm(last_hidden_state)

        # text_embeds.shape = [batch_size, sequence_length, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # eot embedding pos: input_ids.to(dtype=torch.int, device=last_hidden_state.device).argmax(dim=-1)
        # casting to torch.int for onnx compatibility: argmax doesn't support int64 inputs with opset 14
        
        if input_ids is not None:
            eos_embedding_pos = input_ids.to(dtype=torch.int, device=last_hidden_state.device).argmax(dim=-1)
            # print(input_ids, eos_embedding_pos)
        else:
            # pass
            # TODO: is there any exception?
            eos_embedding_pos = torch.tensor([input_embeds.size(1) - 1] * input_embeds.size(0), device=last_hidden_state.device)
        
        pooled_output = last_hidden_state[
            torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
            eos_embedding_pos
        ]

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
        
        
class CLIPModelCanReceiveTextEmbeds(CLIPModel):
    def __init__(self, config: CLIPConfig):
        super().__init__(config)
        
        self.text_model = CLIPTextTransformerCanReceiveEmbed(config.text_config)
        
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        input_embeds: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        return_loss: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        only_return_logits_per_text = False,
        no_grad_text = False
    ) -> Union[Tuple, CLIPOutput]:
        
        # Use CLIP model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if no_grad_text:
            with torch.no_grad():
                text_outputs = self.text_model(
                    input_ids=input_ids,
                    input_embeds=input_embeds,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
        else:
            text_outputs = self.text_model(
                input_ids=input_ids,
                input_embeds=input_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        image_embeds = vision_outputs[1]
        image_embeds = self.visual_projection(image_embeds)

        text_embeds = text_outputs[1]
        text_embeds = self.text_projection(text_embeds)

        # normalized features
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
        logits_per_image = logits_per_text.t()

        if only_return_logits_per_text:
            return logits_per_text
        
        loss = None
        if return_loss:
            loss = clip_loss(logits_per_text)

        if not return_dict:
            output = (logits_per_image, logits_per_text, text_embeds, image_embeds, text_outputs, vision_outputs)
            return ((loss,) + output) if loss is not None else output

        return CLIPOutput(
            loss=loss,
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            text_model_output=text_outputs,
            vision_model_output=vision_outputs,
        )