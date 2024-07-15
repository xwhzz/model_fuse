from transformers import LlamaForCausalLM
import torch
from peft import get_peft_model, LoraConfig, TaskType
from torch import nn
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
    
    def forward(self, x):
        position_ids = torch.cat([torch.arange(x.size(1)).unsqueeze(0)]*x.size(0))
        out = x
        for m in self.model_list:
            out = m(out, position_ids=position_ids)[0]
        return out

decoder_layer_2 = peft_model.base_model.model.model.layers

model_1 = Net(decoder_layer_1[:2])
model_2 = Net(decoder_layer_2[:2])

input_tensor = torch.rand((1,10,4096),dtype=torch.float32)

torch.onnx.export(model_1,
                  input_tensor,
                  "./model/model_1.onnx",
                  export_params=True,
                  opset_version=14,
                  input_names=['input'],
                  output_names=['output'],
                  dynamic_axes={'input': {0: 'batch_size', 1: 'sequence_length'},
                                'output': {0: 'batch_size', 1: 'sequence_length'}})

torch.onnx.export(model_2,
                  input_tensor,
                  "./model/model_2.onnx",
                  export_params=True,
                  opset_version=14,
                  input_names=['input'],
                  output_names=['output'],
                  dynamic_axes={'input': {0: 'batch_size', 1: 'sequence_length'},
                                'output': {0: 'batch_size', 1: 'sequence_length'}})