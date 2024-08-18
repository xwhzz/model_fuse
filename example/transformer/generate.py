from transformers import LlamaForCausalLM
import torch
from peft import get_peft_model, LoraConfig, TaskType
from torch import nn
import os

if not os.path.exists('./org_model'):
    os.makedirs('./org_model')
if not os.path.exists('./model'):
    os.makedirs('./model')

torch.random.manual_seed(0)

model = LlamaForCausalLM.from_pretrained('/data/xwh/CodeLlama-7b-Instruct-hf',use_safetensors=False)

decoder_layer_1 = model.model.layers

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    r=8,
    lora_alpha=32,
    lora_dropout=0.1, 
    bias="none",
)

peft_model = get_peft_model(model, peft_config)

for name, param in peft_model.named_parameters():
    if 'lora' in name:
        torch.nn.init.normal_(param, mean=0.0, std=1.0)

class Net(nn.Module):
    def __init__(self, model_list: nn.ModuleList):
        super().__init__()
        self.model_list = model_list
    
    def forward(self, x, position_ids):
        out = x
        for m in self.model_list:
            out = m(out, position_ids=position_ids)[0]
        return out

decoder_layer_2 = peft_model.base_model.model.model.layers

model_1 = Net(decoder_layer_1[:1])
model_2 = Net(decoder_layer_2[:1])

input_tensor = torch.rand((1,10,4096),dtype=torch.float32)
pos_tensor = torch.arange(10).unsqueeze(0)

torch.onnx.export(model_1,
                  (input_tensor,pos_tensor),
                  "./org_model/model_1.onnx",
                  export_params=True,
                  opset_version=15,
                  input_names=['hidden_states', 'position_ids'],
                  output_names=['output'],
                  dynamic_axes={'hidden_states': {0: 'batch_size', 1: 'sequence_length'},
                                'position_ids': {0: 'batch_size', 1: 'sequence_length'},
                                'output': {0: 'batch_size', 1: 'sequence_length'}})

torch.onnx.export(model_2,
                  (input_tensor,pos_tensor),
                  "./org_model/model_2.onnx",
                  export_params=True,
                  opset_version=15,
                  input_names=['hidden_states', 'position_ids'],
                  output_names=['output'],
                  dynamic_axes={'hidden_states': {0: 'batch_size', 1: 'sequence_length'},
                                'position_ids': {0: 'batch_size', 1: 'sequence_length'},
                                'output': {0: 'batch_size', 1: 'sequence_length'}})