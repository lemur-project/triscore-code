import torch
from torch import nn
from transformers import (
    AutoModel, 
    AutoTokenizer, 
    AutoProcessor, 
    AutoFeatureExtractor,
    PaliGemmaForConditionalGeneration,
    AutoModelForCausalLM
)
from utils.configs import ENCODERS

# IMAGE ENCODERS --------------------

class ImageEncoder(nn.Module):
    def __init__(self, model_name, freeze_model=True):
        super(ImageEncoder, self).__init__()
        self.hidden_size = ENCODERS[model_name]["hidden_size"]
        self.model_id = ENCODERS[model_name]["model_id"]
        self._load_model(model_name)
        self.name = model_name
        if freeze_model:
            for param in self.model.parameters():
                param.requires_grad = False


    def _load_model(self, model_name):
        if model_name in ["DiT", "ViT"]:
            self.model = AutoModel.from_pretrained(self.model_id)
        elif model_name == "PaliGemma":
            paligemma = PaliGemmaForConditionalGeneration.from_pretrained(self.model_id)
            self.model = paligemma.model.vision_tower
            del paligemma
        else:
            raise ValueError(f"Unsupported image encoder model: {model_name}")
        
    def get_processor(self):
        if self.name in ["DiT", "ViT"]:
            return AutoProcessor.from_pretrained(self.model_id)
        elif self.name == "PaliGemma":
            proc_pgemma = AutoProcessor.from_pretrained(self.model_id)
            return proc_pgemma.image_processor
        else:
            raise ValueError(f"Unsupported image encoder model: {self.name}")
    
    def forward_paligemma(self, x):
        # https://discuss.huggingface.co/t/image-embedding-from-paligemma-model/103217
        image_ouputs = self.model(x["pixel_values"])
        x = image_ouputs.last_hidden_state # batch_size, hidden_size, seq_len
        x = x.mean(dim=2) # avg pool over sequence length
        return x # batch_size, hidden_size
    
    def forward_hf(self, **kwargs):
        outputs = self.model(**kwargs, output_hidden_states=True)
        hidden_states = torch.stack(outputs.hidden_states)
        hidden_states = hidden_states.permute(1, 2, 3, 0) # batch_size, seq_len, hidden_size, num_layer
        x = hidden_states[..., -1] # use last layer
        x = x.mean(dim=1) # avg pool over sequence length
        return x

    def forward(self, x):
        if self.name == "PaliGemma":
            return self.forward_paligemma(x)
        else:
            return self.forward_hf(**x)
        
# AUDIO ENCODERS --------------------

class AudioEncoder(nn.Module):
    def __init__(self, model_name, freeze_model=True):
        super(AudioEncoder, self).__init__()
        self.hidden_size = ENCODERS[model_name]["hidden_size"]
        self.model_id = ENCODERS[model_name]["model_id"]
        self._load_model(model_name)
        self.name = model_name
        self.sr = self._get_sampling_rate()

        if freeze_model:
            for param in self.model.parameters():
                param.requires_grad = False

    def _load_model(self, model_name):
        if model_name in ["MERT", "Wav2Vec2"]:
            self.model = AutoModel.from_pretrained(self.model_id, trust_remote_code=True)
        elif model_name == "Whisper":
            whisper_model = AutoModel.from_pretrained(self.model_id)
            self.model = whisper_model.encoder
        else:
            raise ValueError(f"Unsupported audio encoder model: {model_name}") 
        
    def _get_sampling_rate(self):
        if self.name in ["MERT"]:
            return 24_000
        elif self.name in ["Whisper", "Wav2Vec2"]:
            return 16_000
        else:
            raise ValueError(f"Unsupported audio encoder model: {self.name}")

    def get_processor(self):
        if self.name == "MERT":
            return AutoFeatureExtractor.from_pretrained(self.model_id, trust_remote_code=True)
        elif self.name in ["Whisper", "Wav2Vec2"]:
            return AutoProcessor.from_pretrained(self.model_id)
        else:
            raise ValueError(f"Unsupported audio encoder model: {self.name}")

    def forward(self, args):
        outputs = self.model(**args, output_hidden_states=True)
        hidden_states = torch.stack(outputs.hidden_states)
        hidden_states = hidden_states.permute(1, 2, 3, 0) # batch_size, seq_len, hidden_size, num_layer
        x = hidden_states[..., -1] # use last layer
        x = x.mean(dim=1) # avg pool over sequence length
        return x


# SYMBOLIC ENCODERS -------------------- 

class SymbolicEncoder(nn.Module):
    def __init__(self, model_name, freeze_model=True):
        super(SymbolicEncoder, self).__init__()
        self.hidden_size = ENCODERS[model_name]["hidden_size"]
        self.model_id = ENCODERS[model_name]["model_id"]
        self._load_model(model_name)
        self.name = model_name
        self.encoding_input = self._get_symbolic_encoding(model_name)
        if freeze_model:
            for param in self.model.parameters():
                param.requires_grad = False

    def _load_model(self, model_name):
        if model_name in ["RoBERTa", "GraphCodeBERT", "SentenceTransformer"]:
            self.model = AutoModel.from_pretrained(self.model_id)
        else:
            raise ValueError(f"Unsupported symbolic encoder model: {model_name}")
        
    def _get_symbolic_encoding(self, model_name):
        return "kern"
        
    def get_processor(self):
        return AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        
    def forward(self, args):
        outputs = self.model(**args, output_hidden_states=True)
        hidden_states = torch.stack(outputs.hidden_states) 
        # layer to last dimension
        hidden_states = hidden_states.permute(1, 2, 3, 0)
        x = hidden_states[..., -1] # use last layer
        x = x.mean(dim=1) # avg pool over sequence length
        return x