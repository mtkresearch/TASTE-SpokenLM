# TASTE: Text-Aligned Speech Tokenization and Embedding for Spoken Language Modeling

[[Demo](https://mtkresearch.github.io/TASTE-SpokenLM.github.io/)] [[Paper](https://arxiv.org/abs/2504.07053)] [[Model](https://huggingface.co/MediaTek-Research/Llama-1B-TASTE-V0)]  [[Playground](https://www.kaggle.com/code/ycckaggle/playground-taste)]

<b>Liang-Hsuan Tseng*, Yi-Chang Chen*, Kuan-Yi Lee, Da-Shan Shiu, Hung-yi Lee</b><br/>*Equal contribution

Recent efforts target spoken language models (SLMs) that not only listen but also speak for more natural human–LLM interaction. Joint speech–text modeling is a promising direction to achieve this. However, the effectiveness of recent speech tokens for joint modeling remains underexplored. To address this, we introduce <b>T</b>ext-<b>A</b>ligned <b>S</b>peech <b>T</b>okenization and <b>E</b>mbedding (<b>TASTE</b>), a method that directly addresses the modality gap by aligning speech token with the corresponding text transcription during the tokenization stage. We propose a method that can achieve this through a attention-based aggregation mechanism and with speech reconstruction as the training objective. We conduct extensive experiments and show that TASTE can preserve essential paralinguistic information while dramatically reducing the token sequence length. With TASTE, we perform straightforward joint spoken language modeling by using Low-Rank Adaptation on the pre-trained text LLM. Our experimental results show that joint modeling with TASTE and text tokens outperforms other pre-trained SLMs in tasks such as speech continuation and likelihood-based next-speech selection. To our knowledge, TASTE is the first end-to-end approach that utilizes a reconstruction objective to automatically learn a text-aligned speech tokenization and embedding suitable for spoken language modeling.

## Quick Start

Install the `taste_speech` package
```
git clone https://github.com/mtkresearch/TASTE-SpokenLM.git
cd TASTE-SpokenLM
pip install .
```

Install some dependencies,
```
pip install -q torch==2.3.1 torchaudio==2.3.1 torchvision==0.18.1
pip install -q transformers==4.51.1
pip install -q einx==0.3.0 HyperPyYAML==1.2.2 openai-whisper==20231117 onnxruntime-gpu==1.16.0 conformer==0.3.2 lightning==2.2.4
pip install -q datasets matplotlib librosa omegaconf diffusers peft
```

### Inference Completion

```python
from datasets import Dataset
import torchaudio

from taste_speech import TasteConfig, TasteForCausalLM, TasteProcessor

device = 0
model_id = 'MediaTek-Research/Llama-1B-TASTE-V0'
attn_implementation = 'eager'

model = TasteForCausalLM.from_pretrained(model_id, attn_implementation=attn_implementation)

model = model.to(device)
model.eval()

processor = TasteProcessor.from_pretrained(model_id)
generator = processor.get_generator(model_id, device=device)

generate_kwargs = dict(
    llm_tokenizer=processor.llm_tokenizer,
    asr_tokenizer=processor.audio_tokenizer,
    extra_words=8,
    text_top_p=0.3,
    taste_top_p=0.0,
    text_temperature=0.5,
    repetition_penalty=1.1,
)

conditional_audio_paths = ['/path/to/audio.wav']
output_audio_paths = ['/path/to/generated_audio.wav']
sampling_rate = 16000

data = [
    processor(
        audio_path,
        sampling_rate,
        ref_audio_list=[audio_path]
    )
    for audio_path in conditional_audio_paths
]
dataset = Dataset.from_list(data)

for inputs, output_fpath in zip(data, output_audio_paths):
    inputs = {k: inputs[k].to(device) for k in inputs.keys()}
    output = model.inference_completion(
        **inputs,
        conditional_mode='audio',
        **generate_kwargs,
    )
    tts_speech, tts_sr = generator.inference(
        speech_token_ids=output['speech_token_ids'], 
        speech_token_lengths=output['speech_token_lengths'],
        flow_embedding=inputs['speaker_embeds']
    )
    torchaudio.save(output_fpath, tts_speech, tts_sr)
```

### Run Inference

```
python scripts/generate_audio.py --conditional_compl
```

## Training

### Train TASTE Tokenizers

see [STAGE1_TRAIN/](./STAGE1_TRAIN).

### Train TASLM

TBD

## Citation

```
@misc{tseng2025tastetextalignedspeechtokenization,
      title={TASTE: Text-Aligned Speech Tokenization and Embedding for Spoken Language Modeling}, 
      author={Liang-Hsuan Tseng and Yi-Chang Chen and Kuan-Yi Lee and Da-Shan Shiu and Hung-yi Lee},
      year={2025},
      eprint={2504.07053},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2504.07053}, 
}
```

