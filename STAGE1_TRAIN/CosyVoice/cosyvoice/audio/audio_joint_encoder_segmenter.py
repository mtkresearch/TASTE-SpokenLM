# sometimes the encoder and the segmenter are better to be written together. 
import torch
from hyperpyyaml import load_hyperpyyaml
from typing import Dict, Optional, Union, List, Tuple
from torch import nn
import torch.nn.functional as F
import torchaudio
from torch.nn.utils.rnn import pad_sequence, unpad_sequence
from funasr import AutoModel
from funasr.utils.misc import deep_update
import librosa
import logging
import os
import matplotlib.pyplot as plt
import numpy as np
from einops import rearrange
from funasr.frontends.whisper_frontend import WhisperFrontend
from transformers import WhisperTokenizerFast, WhisperFeatureExtractor
from transformers.models.whisper.generation_whisper import _median_filter
from cosyvoice.audio.audio_encoder import WhisperAudioEncoder
from cosyvoice.audio.audio_segmenter import WhisperCrossAttentionSegmenter
from cosyvoice.utils.model_utils import load_whisper_whole_model

# NOTE: Currently, the model is more like a wrapper to prevent re-loading of the same checkpoint or weights for the encoder and segmenter.
# NOTE: you should use `JointEncoderSegmenterAudioTokenizer` in audio_tokenizer.py if you want to use the modules in this file. 
class BaseAudioJointEncoderSegmenter(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    def to(self, device):
        self.device = device
        return super().to(device)
    
    def get_device(self):
        if next(self.parameters(), None) is not None:
            return next(self.parameters()).device
        elif next(self.buffers(), None) is not None:
            return next(self.buffers()).device
        else:
            return 'cpu'
    
    # for the encoder 
    def extract_feature(
        self,
        audio_fpaths: List[str],
    ):
        raise NotImplementedError
    
    def get_audio_encoder(self):
        return self.audio_encoder # should have constructed encoder before calling this
    
    def get_audio_segmenter(self):
        return self.audio_segmenter # should have constructed segmenter before calling this

# One may want to use both WhisperEncoder and WhisperDecoder, as AudioEncoder and AudioSegmenter, respectively.
class WhisperAudioJointEncoderSegmenter(BaseAudioJointEncoderSegmenter):
    def __init__(
        self,
        model_name_or_path: str,
        target_hidden_layer: int = 6,
        attn_implementation: str = "eager",
        dtype: str = 'float32',
        forward_type = "add_and_norm", # Currently support: ['original', add, add_and_norm, asr_attn_pooling]
        make_v_proj_identity: bool = False, 
        is_word_level: bool = False,
        skip_prefix_idx: Optional[int] = None,
        new_vocab: Dict = None,
        **kwargs,
    ): 
        super().__init__()
        whole_model, torch_dtype = load_whisper_whole_model(
            model_name_or_path,
            attn_implementation = attn_implementation,
            dtype = dtype,
            use_custom = True,
        )
        self.attn_implementation = attn_implementation
        self.config = whole_model.config
        self.torch_dtype = torch_dtype
        encoder = whole_model.get_encoder()
        self.audio_encoder = WhisperAudioEncoder(
            model_name_or_path, # for loading the preprocessor
            target_hidden_layer = -1, # extract all including target hidden and the last (for the decoder)
            encoder_model = encoder
        )
        decoder = whole_model.get_decoder()
        self.audio_segmenter = WhisperCrossAttentionSegmenter(
            model_name_or_path,
            decoder_model = decoder,
        )
        self.target_hidden_layer = target_hidden_layer
        # custom module
        # Attempt 1: Add & Norm
        self.forward_type = forward_type
        if self.forward_type == "add_and_norm":
            self.encoder_early_exit_layer_norm = nn.LayerNorm(self.config.d_model) 
        if make_v_proj_identity:
            self._initialize_identity(self.audio_segmenter.decoder.layers[0].encoder_attn.v_proj)
            self._initialize_identity(self.audio_segmenter.decoder.layers[1].encoder_attn.v_proj)
            print("Initialized cross_attn's v_proj with identity matix.")
        self.is_word_level = is_word_level
        if self.is_word_level:
            print("Would adopt word-level averaging")
            assert skip_prefix_idx != None, f"To adopt word level averaging, please set `skip_prefix_idx` properly and with cautious."
        self.skip_prefix_idx = skip_prefix_idx

        # Other tokenizer input
        if new_vocab is not None:
            if new_vocab.get("vocab_size", None) and new_vocab.get("padding_idx", None):
                self.audio_segmenter.decoder.embed_tokens = torch.nn.Embedding(new_vocab["vocab_size"], self.audio_segmenter.decoder.embed_tokens.weight.shape[1], padding_idx=new_vocab["padding_idx"])
            elif new_vocab.get("vocab_size", None):
                self.audio_segmenter.decoder.embed_tokens = torch.nn.Embedding(new_vocab["vocab_size"], self.audio_segmenter.decoder.embed_tokens.weight.shape[1])
            else:
                raise ValueError("Plese provide vocab_size in new_vocab")
    
    def _initialize_identity(self, target_linear_layer):
        # initialize the target_linear_layer as identity matrix
        with torch.no_grad():
            target_linear_layer.weight.copy_(torch.eye(target_linear_layer.in_features))
            target_linear_layer.bias.fill_(0.0)

    # @torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)
    @torch.amp.autocast('cuda')
    def forward(
        self,
        audio_features: torch.Tensor,
        audio_features_lengths: torch.Tensor,
        text_token_for_audio: Optional[torch.Tensor],
        text_token_embed_for_audio: Optional[torch.Tensor], 
        text_token_len: Optional[torch.Tensor],
        *args,
        whisper_text_token: Optional[torch.Tensor] = None, # whisper text tokens with decoder prefix
        whisper_text_token_len: Optional[torch.Tensor] = None,
        words_index: Optional[List[Tuple[int, int, int]]] = None,
        word_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        encoded_results = self.audio_encoder(
            audio_features,
            audio_features_lengths,
            output_hidden_states=self.target_hidden_layer,
        )
        encoded_feats, encoded_feats_len = encoded_results['encoded_feats'], encoded_results['encoded_feats_lengths']
        last_encoder_hidden, target_encoder_hidden = encoded_feats['last_hidden'], encoded_feats[f'{self.target_hidden_layer}']
        # Attempt 1: direct Add & (Norm?)
        if self.forward_type == "original":
            # for debug
            encoded_feats = last_encoder_hidden
            print(f"Encoder last hidden: {last_encoder_hidden[0][:5,:5]}")
            print(f"Encoder target hidden: {target_encoder_hidden[0][:5,:5]}")
            _encoder_results_for_test = self.audio_encoder.encoder(
                input_features = audio_features.transpose(1, 2),
                output_hidden_states = True,
                return_dict = True,
            )
            _encoder_last_hidden_for_test = _encoder_results_for_test.last_hidden_state
            _encoder_target_hidden_for_test = _encoder_results_for_test.hidden_states[self.target_hidden_layer]
            print(f"Encoder last hidden (orig): {_encoder_last_hidden_for_test[0][:5,:5]}")
            print(f"Encoder target hidden (orig): {_encoder_target_hidden_for_test[0][:5,:5]}")
        elif "add" in self.forward_type:
            encoded_feats = last_encoder_hidden + target_encoder_hidden
            if self.forward_type == "add_and_norm":
                encoded_feats = self.encoder_early_exit_layer_norm(encoded_feats) #(B, T, C)
        elif self.forward_type == "asr_attn_pooling":
            encoded_feats = {
                "states_for_key": last_encoder_hidden,
                "states_for_val": target_encoder_hidden,
            }
        ## decoder forward
        output_attentions = (self.attn_implementation == "eager")
        decoder_outputs = self.audio_segmenter.decoder(
            input_ids = whisper_text_token,
            encoder_hidden_states = encoded_feats,
            output_attentions = output_attentions,
        )

        decoder_last_hidden_state = decoder_outputs.last_hidden_state
        decoder_last_hidden_state_len = whisper_text_token_len

        if self.skip_prefix_idx != None:
            decoder_last_hidden_state = decoder_last_hidden_state[:, self.skip_prefix_idx:, :] # skip the offset of prefix
            # decoder_last_hidden_state_len -= self.skip_prefix_idx # reduce output len based on skip_prefix_idx
            decoder_last_hidden_state_len = decoder_last_hidden_state_len - self.skip_prefix_idx # reduce output len based on skip_prefix_idx. avoid in-place for safety

        if self.is_word_level:
            if words_index == None:
                assert word_ids != None, "joint encoder segmenter is word-level, please pass `words_index` or `word_ids` properly!"
                words_index = self._convert_word_ids_to_words_index(word_ids, decoder_last_hidden_state_len)
            
            decoder_last_hidden_state = self._averaging_subword_to_word_level(decoder_last_hidden_state, words_index)

            # b, t1, t2 = words_index[-1]
            # _decoder_hidden_slices = decoder_last_hidden_state[b, t1:t2, :]
            # print(_decoder_hidden_slices)
            # print(_decoder_hidden_slices.shape)

        segmented_results = {
            'segmented_feats': decoder_last_hidden_state,
            'segmented_feats_lengths': decoder_last_hidden_state_len,
            'decoder_outputs': decoder_outputs,
        }
        encoded_results['encoded_feats'] = encoded_feats

        return encoded_results, segmented_results
    
    def _averaging_subword_to_word_level(
        self,
        features: Optional[torch.Tensor] = None, # in the shape of (B, T, C)
        words_index: Optional[List[Tuple[int, int, int]]] = None,
    ):
        bsz, tsz, csz = features.shape
        # iterate through segments with more than one subword 
        averaged_features = features.clone()
        for (b, t1, t2) in words_index:
            if b >= bsz or t1 < 0 or t2 > tsz or t1 >= t2:
                raise ValueError("Invalid segment indices")
            # extract segment and compute the mean along the time dimension
            segment_mean = features[b, t1:t2, :].mean(dim=0, keepdim=True) # Shape: (1, C)
            # assign the mean value back to features
            averaged_features[b, t1:t2, :] = segment_mean # NOTE: Avoid in-place operation for back-propogation.
        
        return averaged_features
    
    def _convert_word_ids_to_words_index(
        self,
        word_ids,
        token_lengths,
    ):  
        words_index = []
        for b, (word_id, token_len) in enumerate(zip(word_ids, token_lengths)):
            _, counts = word_id.unique_consecutive(return_counts=True)
            _valid_pooling_lens_mask = counts > 1
            _accumulate_counts = counts.cumsum(-1)
            _valid_lens_mask = _accumulate_counts <= token_len
            mask = _valid_pooling_lens_mask.logical_and_(_valid_lens_mask)
            valid_end_idx = _accumulate_counts[mask]
            _valid_start_idx = _accumulate_counts[:-1][mask[1:]]
            valid_start_idx = torch.zeros_like(valid_end_idx)
            if len(valid_start_idx) > len(_valid_start_idx):
                valid_start_idx[1:] = _valid_start_idx
            else:
                valid_start_idx = _valid_start_idx
            for s, e in zip(valid_start_idx, valid_end_idx):
                words_index.append((b, s.item(), e.item()))
        return words_index

    def _get_alignment_map(
        self,
        generate_outputs,
        alignment_heads,
        time_precision=0.02,
        num_frames=None,
        use_orig=False,
        median_filter_width=7,
    ):  
        if use_orig:
            cross_attentions = []
            for i in range(self.config.decoder_layers):
                cross_attentions.append(torch.cat([x[i] for x in generate_outputs.cross_attentions], dim=2))
        else:
            # print(generate_outputs.cross_attentions, generate_outputs.cross_attentions[0].shape, len(generate_outputs.cross_attentions))
            cross_attentions = generate_outputs.cross_attentions
        # Select specific cross-attention layers and heads. This is a tensor
        # of shape (batch size, num selected, output length, input length).
        weights = torch.stack([cross_attentions[l][:, h] for l, h in alignment_heads])
        # print(weights.shape)
        weights = weights.permute([1, 0, 2, 3])

        # print(weights)
        # print(weights.shape)
        # normalize and smoothen the weights
        std = torch.std(weights, dim=-2, keepdim=True, unbiased=False)
        mean = torch.mean(weights, dim=-2, keepdim=True)
        weights = (weights - mean) / std
        weights = _median_filter(weights, median_filter_width)

        weights = weights.mean(dim=1)

        matrixes = []
        batch_size = weights.size(0)
        for batch_idx in range(batch_size):
            matrix = weights[batch_idx]
            matrix = matrix.cpu().double().numpy()
            matrixes.append(matrix)
        return matrixes


def draw_attn_map(attn_map, output_fpath, token_ids):
    x_size, y_size = attn_map.shape
    plt.figure(figsize=(10, 8))
    plt.imshow(attn_map.T, aspect='auto', origin='lower', interpolation='none', cmap='viridis')
    plt.xlabel('whisper subwords')
    plt.xticks(ticks=np.arange(x_size), labels=token_ids)

    plt.ylabel('Encoder Last Hidden')
    plt.yticks(ticks=np.arange(y_size))

    plt.colorbar(label='Attention Weight (normalized)')
    plt.savefig(f"{output_fpath}")

def test_whisper_joint_encoder_segmenter():
    RTSLM_WORK_DIR = os.getenv(
        'RTSLM_WORK_DIR'
    )
    model_name_or_path = "/proj/mtklmadm/dev/mtk53678/rtslm_storage/pretrained_models/distil-whisper-large-v3"
    whisper_joint_encoder_segmenter = WhisperAudioJointEncoderSegmenter(
        model_name_or_path,
        attn_implementation="flash_attention_2",
        dtype="bfloat16",
        forward_type="asr_attn_pooling",
        make_v_proj_identity=False,
    )
    print(whisper_joint_encoder_segmenter.state_dict().keys())

    audio_fpaths = [
        # f"{RTSLM_WORK_DIR}/CosyVoice/en.mp3",
        # f"{RTSLM_WORK_DIR}/CosyVoice/instruct4.wav",
        "/proj/mtklmadm/data/speech/LibriTTS/LibriTTS/test-clean/7127/75946/7127_75946_000033_000000.wav"	
    ]
    audio_feat, audio_feat_len = whisper_joint_encoder_segmenter.audio_encoder.extract_feature(
        audio_fpaths,
    )
    text_tokenizer = WhisperTokenizerFast.from_pretrained(model_name_or_path)
    test_text = [
        "I reserve your services for a better occasion and believe me, they will only be the better appreciated."
    ]
        
    whisper_text_token = text_tokenizer(test_text).input_ids
    whisper_text_token = [torch.tensor(text_token) for text_token in whisper_text_token]
    whisper_text_token_len = torch.tensor([len(t) for t in whisper_text_token])
    whisper_text_token = pad_sequence(whisper_text_token, batch_first=True, padding_value=text_tokenizer.pad_token_id)
    print(audio_feat_len, whisper_text_token, whisper_text_token_len)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    whisper_joint_encoder_segmenter.to(device)
    whisper_joint_encoder_segmenter.eval()
    with torch.no_grad():
        encoded_results, segmented_results = whisper_joint_encoder_segmenter(
            audio_feat.to(device),
            audio_feat_len.to(device),
            None, None, None,
            whisper_text_token = whisper_text_token.to(device),
            whisper_text_token_len = whisper_text_token_len.to(device),
        )
    print(encoded_results['encoded_feats'])
    print(segmented_results['segmented_feats'])

def test_alignment():
    RTSLM_WORK_DIR = os.getenv(
        'RTSLM_WORK_DIR'
    )
    model_name_or_path = "/proj/mtklmadm/dev/mtk53678/rtslm_storage/pretrained_models/distil-whisper-large-v3"
    whisper_joint_encoder_segmenter = WhisperAudioJointEncoderSegmenter(
        model_name_or_path,
        attn_implementation="eager",
        dtype="float16",
        forward_type="asr_attn_pooling",
        make_v_proj_identity=True,
    )

    exp_root = f"{RTSLM_WORK_DIR}/CosyVoice/examples/libritts/cosyvoice/exp/cosyvoice/llm/torch_ddp"
    exp_name = f"whisper_asr-attn-pooling_vq-straight_text-pretrained_vproj-eye_use-asr_skip-special-early_word"
    exp_dir = os.path.join(exp_root, exp_name)
    config_fpath = os.path.join(exp_dir, "config.yaml")
    ckpt_fpath = os.path.join(exp_dir, "epoch_46_whole.pt")
    with open(config_fpath, 'r') as f:
        print('Loading config')
        config = load_hyperpyyaml(f)
    
    _state_dict = torch.load(ckpt_fpath, map_location='cpu')
    _llm = config['llm']
    _llm.load_state_dict(_state_dict)
    whisper_joint_encoder_segmenter.load_state_dict(_llm.audio_tokenizer.audio_joint_encoder_segmenter.state_dict())

    audio_fpaths = [
        # f"{RTSLM_WORK_DIR}/CosyVoice/instruct4.wav",
        "/proj/mtklmadm/data/speech/LibriTTS/LibriTTS/test-clean/7127/75946/7127_75946_000033_000000.wav"	
    ]
    audio_feat, audio_feat_len = whisper_joint_encoder_segmenter.audio_encoder.extract_feature(
        audio_fpaths,
    )

    text_tokenizer = WhisperTokenizerFast.from_pretrained(model_name_or_path)
    test_text = [
        "I reserve your services for a better occasion and believe me, they will only be the better appreciated."
    ]
        
    whisper_text_token_ids = text_tokenizer(test_text).input_ids
    whisper_text_token_ids = [text_token[:-1] for text_token in whisper_text_token_ids] # remove eos
    # print(whisper_text_token_ids)
    whisper_text_token = [torch.tensor(text_token[:]) for text_token in whisper_text_token_ids]
    whisper_text_token_len = torch.tensor([len(t) for t in whisper_text_token])
    whisper_text_token = pad_sequence(whisper_text_token, batch_first=True, padding_value=text_tokenizer.pad_token_id)
    # print(audio_feat_len, whisper_text_token, whisper_text_token_len)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    whisper_joint_encoder_segmenter.to(device)
    whisper_joint_encoder_segmenter.eval()
    # whole_model.to(device)
    # whole_model.eval()
    with torch.no_grad():
        encoded_results, segmented_results = whisper_joint_encoder_segmenter(
            audio_feat.to(device),
            audio_feat_len.to(device),
            None, None, None,
            whisper_text_token = whisper_text_token.to(device),
            whisper_text_token_len = whisper_text_token_len.to(device),
        )
        # original_outputs = whole_model(
        #     input_features = audio_feat.transpose(1, 2).to(device),
        #     decoder_input_ids = whisper_text_token.to(device),
        #     output_attentions = True
        # )
        # orig_encoder_last_hidden = original_outputs.encoder_last_hidden_state
        # print(f"Orig encoder last hidden: {orig_encoder_last_hidden[0][:5,:5]}")

    generate_outputs = segmented_results['decoder_outputs']

    # test getting alignment
    exp_res_root = f"{RTSLM_WORK_DIR}/CosyVoice/examples/libritts/cosyvoice/exp_result"
    exp_res_dir = os.path.join(exp_res_root, exp_name)
    output_dir = os.path.join(exp_res_dir, "attn_map")
    os.makedirs(output_dir, exist_ok=True)
    for layer in [0, 1]:
        alignment_heads = [[layer, i] for i in range(20)]
        alignment_maps = whisper_joint_encoder_segmenter._get_alignment_map(generate_outputs, alignment_heads)
        for i, alignment_map in enumerate(alignment_maps):
            fname = f'attn_map-{i}_layer-{layer}'
            output_fpath = os.path.join(output_dir, fname)
            draw_attn_map(alignment_map, output_fpath, whisper_text_token_ids[i])
    
    # orig_alignment_maps = whisper_joint_encoder_segmenter._get_alignment_map(original_outputs, alignment_heads)
    # output_dir = './.local'
    # for i, alignment_map in enumerate(orig_alignment_maps):
    #     fname = f'orig_attn_map-{i}_layer-{layer}'
    #     output_fpath = os.path.join(output_dir, fname)
    #     draw_attn_map(alignment_map, output_fpath, whisper_text_token_ids[i])
        

def test_feature_extraction():
    RTSLM_WORK_DIR = os.getenv(
        'RTSLM_WORK_DIR'
    )
    model_name_or_path = "/proj/mtklmadm/dev/mtk53678/rtslm_storage/pretrained_models/distil-whisper-large-v3"
    whisper_joint_encoder_segmenter = WhisperAudioJointEncoderSegmenter(
        model_name_or_path,
        attn_implementation="eager",
        dtype="float16",
        forward_type="original",
        make_v_proj_identity=False,
    )

    audio_fpaths = [
        # f"{RTSLM_WORK_DIR}/CosyVoice/instruct4.wav",
        "/proj/mtklmadm/data/speech/LibriTTS/LibriTTS/test-clean/7127/75946/7127_75946_000033_000000.wav"	
    ]
    audio_feat, audio_feat_len = whisper_joint_encoder_segmenter.audio_encoder.extract_feature(
        audio_fpaths,
    )
    print(audio_feat.shape)
    print(f"Custom audio: {audio_feat[0][-5:, :5]}, shape={audio_feat.shape}, feat_len={audio_feat_len}")
    # ---------------------------------- #
    audio_inputs = [librosa.load(audio_fpath, sr=16_000)[0] for audio_fpath in audio_fpaths]
    audio_inputs_len = [len(audio_input) for audio_input in audio_inputs]
    # test original whisper extractor
    whisper_extractor = whisper_joint_encoder_segmenter.audio_encoder.processor.feature_extractor
    _orig_audio_feat = whisper_extractor(
        audio_inputs,
    )
    orig_audio_feat = torch.tensor(_orig_audio_feat['input_features']).transpose(1, 2)
    custom_orig_hits = (audio_feat == orig_audio_feat).sum()
    print(f"Orig audio: {orig_audio_feat[0][-5:, :5]}, shape={orig_audio_feat.shape}")
    print(f"Custom vs Orig Hits: {custom_orig_hits}")
    # test funasr whisper frontend
    funasr_whisper_frontend = WhisperFrontend(
        whisper_model = 'large-v3',
        do_pad_trim = True,
        permute = True,
    )
    funasr_audio_feat, funasr_audio_feat_len = funasr_whisper_frontend(
        torch.tensor(np.array(audio_inputs)),
        torch.tensor(audio_inputs_len),
    )

    print(f"Funasr audio: {funasr_audio_feat[0][-5:, :5]}, shape={funasr_audio_feat.shape}, feat_len={funasr_audio_feat_len}")
    funasr_orig_hits = (orig_audio_feat == funasr_audio_feat).sum()
    print(f"Funasr vs Orig Hits: {funasr_orig_hits}")
    diff_abs_sum = (funasr_audio_feat - orig_audio_feat).abs().sum()
    print(diff_abs_sum)

if __name__ == "__main__":
    # test_whisper_joint_encoder_segmenter()
    test_alignment()
    # test_feature_extraction()