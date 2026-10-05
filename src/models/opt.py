from transformers import AutoModelForCausalLM
import torch
from .base import BaseModelWrapper

class OPTWrapper(BaseModelWrapper):
    def __init__(self, model_name, tokenizer, batch_size, seqlen, device, dtype):
        super().__init__(model_name, tokenizer, batch_size, seqlen, device, dtype)
        self.layer_prefix = "model.decoder.layers"
        self.num_layers = len(self.model.model.decoder.layers)

    def get_mlp_input(self, layer_input):
        current_layer = self.get_layer_module(self.current_layer_idx)
        num_samples = layer_input.shape[0]
        num_batches = (num_samples + self.batch_size - 1) // self.batch_size

        additional_layer_inputs = {"attention_mask": None}
        for k, v in self.kwargs.items():
            additional_layer_inputs[k] = v

        for batch_idx in range(num_batches):
            start, end = batch_idx * self.batch_size, min((batch_idx + 1) * self.batch_size, num_samples)
            batch = layer_input[start:end].to(self.device)
            if current_layer.do_layer_norm_before:
                hidden_states = current_layer.self_attn_layer_norm(batch)

            hidden_states, _ = current_layer.self_attn(hidden_states, **additional_layer_inputs)
            hidden_states = torch.nn.functional.dropout(hidden_states, p=current_layer.dropout, training=current_layer.training)
            hidden_states = batch + hidden_states

            if not current_layer.do_layer_norm_before:
                hidden_states = current_layer.self_attn_layer_norm(hidden_states)

            layer_input[start:end] = hidden_states.detach().cpu()
    
    def get_mlp_output(self, mlp_input_batch):
        current_layer = self.get_layer_module(self.current_layer_idx)

        mlp_input_batch_shape = mlp_input_batch.shape
        hidden_states = mlp_input_batch.reshape(-1, mlp_input_batch.size(-1))
        if current_layer.do_layer_norm_before:
            hidden_states = current_layer.final_layer_norm(hidden_states)

        hidden_states = current_layer.fc1(hidden_states)
        hidden_states = current_layer.activation_fn(hidden_states)

        hidden_states = current_layer.fc2(hidden_states)
        hidden_states = torch.nn.functional.dropout(hidden_states, p=current_layer.dropout, training=current_layer.training)

        hidden_states = hidden_states.view(mlp_input_batch_shape) + mlp_input_batch

        if not current_layer.do_layer_norm_before:
            hidden_states = current_layer.final_layer_norm(hidden_states)

        return hidden_states

    def get_layer_module(self, idx):
        return self.model.model.decoder.layers[idx]
    
    def move_embed_to(self, device):
        self.model.model.decoder.embed_tokens = self.model.model.decoder.embed_tokens.to(device)
        self.model.model.decoder.embed_positions = self.model.model.decoder.embed_positions.to(device)
