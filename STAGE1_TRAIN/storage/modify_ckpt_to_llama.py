import torch
import sys
import os
from collections import OrderedDict

# Add CosyVoice to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../CosyVoice'))

from cosyvoice.llm.llm import TransformerLM
from cosyvoice.transformer.encoder import ConformerEncoder, TransformerEncoder

def print_module_info(model):
    print("\nDetailed Model Architecture:")
    for name, module in model.named_modules():
        if len(name) > 0:  # Skip the root module
            print(f"\nModule: {name}")
            print(f"Type: {type(module).__name__}")
            
            # Get parameters
            params = sum(p.numel() for p in module.parameters())
            print(f"Parameters: {params:,}")
            
            # Try to get input/output dimensions if available
            if hasattr(module, 'in_features'):
                print(f"Input features: {module.in_features}")
            if hasattr(module, 'out_features'):
                print(f"Output features: {module.out_features}")
            if hasattr(module, 'in_channels'):
                print(f"Input channels: {module.in_channels}")
            if hasattr(module, 'out_channels'):
                print(f"Output channels: {module.out_channels}")
            if hasattr(module, 'input_size'):
                print(f"Input size: {module.input_size}")
            if hasattr(module, 'output_size'):
                print(f"Output size: {module.output_size}")

def main():
    # Load checkpoint
    checkpoint_path = './checkpoints/text-only_baseline/checkpoint_best.pt'
    print(f"Loading checkpoint from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Print checkpoint keys
    print("\nCheckpoint keys:")
    for key in checkpoint.keys():
        if isinstance(checkpoint[key], dict):
            print(f"{key}:")
            for subkey in checkpoint[key].keys():
                print(f"  {subkey}")
        else:
            print(key)
    
    # Create model according to config
    model = TransformerLM(
        text_encoder_input_size=512,
        llm_input_size=1024,
        llm_output_size=1024,
        text_token_size=128256,  # Modified to get desired embedding shape
        speech_token_size=4096,
        length_normalized_loss=True,
        lsm_weight=0,
        spk_embed_dim=192,
        text_encoder=ConformerEncoder(
            input_size=512,
            output_size=1024,
            attention_heads=8,
            linear_units=2048,
            num_blocks=3,
            dropout_rate=0.1,
            positional_dropout_rate=0.1,
            attention_dropout_rate=0,
            normalize_before=True,
            input_layer='linear',
            pos_enc_layer_type='rel_pos_espnet',
            selfattention_layer_type='rel_selfattn',
            use_cnn_module=False,
            macaron_style=False,
            use_dynamic_chunk=False,
            use_dynamic_left_chunk=False,
            static_chunk_size=1
        ),
        llm=TransformerEncoder(
            input_size=1024,
            output_size=1024,
            attention_heads=8,
            linear_units=2048,
            num_blocks=7,
            dropout_rate=0.1,
            positional_dropout_rate=0.1,
            attention_dropout_rate=0,
            input_layer='linear_legacy',
            pos_enc_layer_type='rel_pos_espnet',
            selfattention_layer_type='rel_selfattn',
            static_chunk_size=1
        )
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    state_dict = checkpoint
    
    # Create new text embedding weight with desired shape
    print("\nCreating new text embedding weight with shape [128256, 512]")
    new_embedding = torch.zeros(128256, 512)
    # Initialize with xavier uniform for better starting point
    torch.nn.init.xavier_uniform_(new_embedding)
    
    # Replace text embedding in state dict
    state_dict['text_embedding.weight'] = new_embedding
    
    # Load state dict with new embedding
    model.load_state_dict(state_dict)
    print("\nModel loaded successfully!")
    print_module_info(model)
        
    # Print state dict keys to see all available tensors
    print("\nModel State Dict Keys and Shapes:")
    for key, tensor in model.state_dict().items():
        print(f"{key}: {tensor.shape}")
    
    # Save the modified model
    output_path = './checkpoints/text-only_baseline/checkpoint_llama.pt'
    print(f"\nSaving modified model to {output_path}")
    torch.save(model.state_dict(), output_path)
    print("Model saved successfully!")

if __name__ == "__main__":
    main()