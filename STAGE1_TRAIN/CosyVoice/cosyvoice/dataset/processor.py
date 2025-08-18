# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import random

import torch
import torchaudio
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq
import torch.nn.functional as F

# from cosyvoice.utils.audio_utils import ResamplerDict
from datasets import Dataset
from io import BytesIO
from torch.nn.utils.rnn import pad_sequence
from funasr.utils.load_utils import extract_fbank
from transformers import WhisperTokenizerFast

# torchaudio.set_audio_backend('soundfile')

AUDIO_FORMAT_SETS = set(['flac', 'mp3', 'm4a', 'ogg', 'opus', 'wav', 'wma'])


def parquet_opener(data, mode='train', tts_data={}):
    """ Give url or local file, return file descriptor
        Inplace operation.

        Args:
            data(Iterable[str]): url or local file list

        Returns:
            Iterable[{src, stream}]
        
        each `sample` would be something like: {'src': 'data/train-clean-100/parquet/parquet_000000002.tar', 'rank': 0, 'world_size': 1, 'worker_id': 0, 'num_workers': 1}
        each row in df: (8 columns currently)
        utt                                        908_31957_000023_000002
        wav              /root/rtslm/work/corpus/rtslm/libritts/LibriTT...
        audio_data       b'RIFF...
        text             Oh, to shoot My soul's full meaning into futur...
        spk                                                            908
        utt_embedding    [-1.479232907295227, 0.5104239583015442, 0.136...
        spk_embedding    [-1.3228259086608887, 0.9402310252189636, 0.47...
        speech_token     [561, 262, 1089, 213, 471, 59, 421, 590, 649, ...
    """
    for sample in data:
        assert 'src' in sample
        url = sample['src']
        try:
            if ".arrow" in url:
                # use hf to load .arrow-style data
                url = url.split(' ')[0]
                # using hf would cause errors
                ds = Dataset.from_file(url)
                for idx, _sample in enumerate(ds):
                    # if idx == 200: break
                    _speech_pt = torch.tensor(_sample['mp3']['array'], dtype=torch.float32)
                    if _speech_pt.dim() == 1:
                        _speech_pt = _speech_pt.unsqueeze(0)
                    _key = url.split('.')[0].split('/')[-1] + "__" + _sample['__key__']
                    _new_sample = {
                        'src': url, 
                        'utt':  _key,
                        'wav':  _sample['mp3']['path'],
                        'text': _sample['json']['text'],
                        'utt_embedding': _sample['spk_emb'],
                        'spk_embedding': _sample['spk_emb'],
                        'speech_token': _sample['s3_token'],
                        'audio_data': None,
                        # audio data here is different from the ones in the parquet file. Here they are directly to be in the hf audio form 
                        # _sample['mp3']: {'path': `path`, 'sampling_rate': 24000, 'array': np.array(1D, dtype=float64)}.
                        'speech': _speech_pt,
                        'sample_rate': _sample['mp3']['sampling_rate']
                    }
                    yield _new_sample
            else:
                # load parquet
                df = pq.read_table(url).to_pandas()
                for i in range(len(df)):
                    if mode == 'inference' and df.loc[i, 'utt'] not in tts_data:
                        continue
                    sample.update(dict(df.loc[i]))
                    if mode == 'train':
                        # NOTE do not return sample directly, must initialize a new dict
                        yield {**sample}
                    else:
                        for index, text in enumerate(tts_data[df.loc[i, 'utt']]):
                            yield {**sample, 'tts_index': index, 'tts_text': text}
        except Exception as ex:
            logging.warning('Failed to open {}, ex info {}'.format(url, ex))

def filter(data,
           max_length=10240,
           min_length=10,
           token_max_length=200,
           token_min_length=1,
           min_output_input_ratio=0.0005,
           max_output_input_ratio=1,
           mode='train'):
    """ Filter sample according to feature and label length
        Inplace operation.

        Args::
            data: Iterable[{key, wav, label, sample_rate}]
            max_length: drop utterance which is greater than max_length(10ms)
            min_length: drop utterance which is less than min_length(10ms)
            token_max_length: drop utterance which is greater than
                token_max_length, especially when use char unit for
                english modeling
            token_min_length: drop utterance which is
                less than token_max_length
            min_output_input_ratio: minimal ration of
                token_length / feats_length(10ms)
            max_output_input_ratio: maximum ration of
                token_length / feats_length(10ms)

        Returns:
            Iterable[{key, wav, label, sample_rate}]
    """
    for sample in data:
        if sample.get('sample_rate', None) == None:
            sample['speech'], sample['sample_rate'] = torchaudio.load(BytesIO(sample['audio_data']))
            # otherwise has already been loaded.
        if sample.get('audio_data', False):
            del sample['audio_data']
        # sample['wav'] is torch.Tensor, we have 100 frames every second
        num_frames = sample['speech'].size(1) / sample['sample_rate'] * 100
        if num_frames < min_length:
            continue
        if num_frames > max_length:
            continue
        if len(sample['text_token']) < token_min_length:
            continue
        if len(sample['text_token']) > token_max_length:
            continue
        if len(sample['speech_token']) == 0:
            continue
        if num_frames != 0:
            if len(sample['text_token']) / num_frames < min_output_input_ratio:
                continue
            if len(sample['text_token']) / num_frames > max_output_input_ratio:
                continue
        yield sample


def resample(data, resample_rate=22050, min_sample_rate=16000, mode='train'):
    """ Resample data.
        Inplace operation.

        Args:
            data: Iterable[{key, wav, label, sample_rate}]
            resample_rate: target resample rate

        Returns:
            Iterable[{key, wav, label, sample_rate}]
    """
    # resampler_dict = ResamplerDict()
    for sample in data:
        assert 'sample_rate' in sample
        assert 'speech' in sample
        sample_rate = sample['sample_rate']
        waveform = sample['speech']
        if sample_rate != resample_rate:
            if sample_rate < min_sample_rate:
                continue
            sample['sample_rate'] = resample_rate
            sample['speech'] = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=resample_rate)(waveform)
        max_val = sample['speech'].abs().max()
        if max_val > 1:
            sample['speech'] /= max_val
        yield sample

# NOTE: This is the audio feature extraction process for our `audio branch`!
def extract_audio(data, audio_extractor, mode='train', target_sample_rate=16_000, **kwargs):
    """ Extract audio for audio branch
        Args:
            data: Iterable[{key, wav, label, sample_rate}]

        Returns:
            Iterable[{key, feat, label}]
    """
    audio_extractor.eval()
    with torch.no_grad():
        for sample in data:
            assert 'sample_rate' in sample
            assert 'speech' in sample
            waveform, orig_sample_rate = sample['speech'], sample['sample_rate']
            feat_len = None
            if orig_sample_rate != target_sample_rate:
                waveform = torchaudio.transforms.Resample(
                    orig_freq=orig_sample_rate, new_freq=target_sample_rate)(waveform).mean(0)
            waveform_length = [waveform.shape[-1]]
            feat, feat_len = audio_extractor(waveform.view(1,-1), waveform_length, **kwargs)
            sample['audio_feat'] = feat.squeeze(dim=0)
            sample['audio_feat_len'] = feat_len
            yield sample

def compute_fbank(data,
                  feat_extractor,
                  mode='train'):
    """ Extract fbank

        Args:
            data: Iterable[{key, wav, label, sample_rate}]

        Returns:
            Iterable[{key, feat, label}]
    """
    for sample in data:
        assert 'sample_rate' in sample
        assert 'speech' in sample
        assert 'utt' in sample
        assert 'text_token' in sample
        waveform = sample['speech']
        mat = feat_extractor(waveform).squeeze(dim=0).transpose(0, 1)
        sample['speech_feat'] = mat
        del sample['speech']
        yield sample


def parse_embedding(data, normalize, mode='train'):
    """ Parse utt_embedding/spk_embedding

        Args:
            data: Iterable[{key, wav, label, sample_rate}]

        Returns:
            Iterable[{key, feat, label}]
    """
    for sample in data:
        sample['utt_embedding'] = torch.tensor(sample['utt_embedding'], dtype=torch.float32)
        sample['spk_embedding'] = torch.tensor(sample['spk_embedding'], dtype=torch.float32)
        if normalize:
            sample['utt_embedding'] = F.normalize(sample['utt_embedding'], dim=0)
            sample['spk_embedding'] = F.normalize(sample['spk_embedding'], dim=0)
        yield sample


def tokenize(data, get_tokenizer, allowed_special, mode='train', use_asr_text=False):
    """ Decode text to chars or BPE
        Inplace operation

        Args:
            data: Iterable[{key, wav, txt, sample_rate}]

        Returns:
            Iterable[{key, wav, txt, tokens, label, sample_rate}]
    """
    tokenizer = get_tokenizer()
    if use_asr_text:
        logging.debug(f"Will use text from preasr!")
    for sample in data:
        assert 'text' in sample
        if use_asr_text:
            words = [d['word'] for d in sample['asr_chunks']]
            text = ''.join(words).strip()
            sample['asr_text'] = text
        else:
            text = sample['text']
        sample['text_token'] = tokenizer.encode(text, allowed_special=allowed_special)
        if mode == 'inference':
            sample['tts_text_token'] = tokenizer.encode(sample['tts_text'], allowed_special=allowed_special)
        yield sample

def tokenize_whisper(data, tokenizer_name_or_fpath, task='transcribe', language='en', no_timestamps=True, add_bos=False, add_eos=False, mode='train', use_asr_text=False, overwrite_text_token=False, use_wrapped=False):
    """ Decode text to chars or BPE
        Inplace operation

        Args:
            data: Iterable[{key, wav, txt, sample_rate}]

        Returns:
            Iterable[{key, wav, txt, tokens, label, sample_rate}]
    """
    if "distil-large-v3" in tokenizer_name_or_fpath or "whisper" in tokenizer_name_or_fpath:
        tokenizer = WhisperTokenizerFast.from_pretrained(
            tokenizer_name_or_fpath,
        )
        forced_decoder_ids = tokenizer.get_decoder_prompt_ids(
            task = task,
            language = language,
            no_timestamps = no_timestamps,
        )
        _prefix_tokens = tokenizer.prefix_tokens
        prefix_token_to_wrap  = _prefix_tokens if add_bos else _prefix_tokens[1:]
        postfix_token_to_wrap = [tokenizer.eos_token_id] if add_eos else []
        _skip_prefix_idx = len(prefix_token_to_wrap)
        logging.info(f"Tokenizer is from transformers `WhisperTokenizerFast` of transformers. Decoder prefix ids: {forced_decoder_ids}.")
    # If not whisper, apply this path
    else:
        # Using Llama tokenizer instead of Whisper
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_fpath)
        
        # Llama tokenizer setup
        prefix_token_to_wrap = [tokenizer.bos_token_id] if add_bos else []
        postfix_token_to_wrap = [tokenizer.eos_token_id] if add_eos else []
        _skip_prefix_idx = len(prefix_token_to_wrap)
        logging.info(f"Using Llama tokenizer from {tokenizer_name_or_fpath}")
    
    if use_asr_text:
        logging.debug(f"Will use text from preasr!")
    assert not (use_wrapped and overwrite_text_token and not add_eos), f"Using wrapped and overwriting previous text token without add_eos at the same time is weired."
    if use_wrapped:
        logging.info(f"Will directly wrap the token!, prefix={prefix_token_to_wrap}, postfix={postfix_token_to_wrap}")
        if use_asr_text:
            logging.debug(f"`use_asr_text` is set to True. Please ensure the text_token parsed previously is from preasr.")
    for sample in data:
        if use_wrapped:
            text_token = sample['text_token']
            input_ids = prefix_token_to_wrap + text_token + postfix_token_to_wrap
        else:
            if use_asr_text:
                words = [d['word'] for d in sample['asr_chunks']]
                text = ''.join(words).strip()
                sample['asr_text'] = text
            else:
                text = sample['text']
            input_ids = tokenizer(text).input_ids # NOTE: it's list instead of np array, but it's okay since in padding we will convert it to tensor.
            if not add_bos:
                input_ids = input_ids[1:]
            if not add_eos:
                input_ids = input_ids[:-1]
        if overwrite_text_token:
            sample['text_token'] = input_ids[_skip_prefix_idx:]
        sample['whisper_text_token'] = input_ids
        if mode == 'inference':
            sample['tts_whisper_text_token'] = input_ids
            if overwrite_text_token:
                sample['tts_text_token'] = input_ids[_skip_prefix_idx:]
        yield sample

def tokenize_by_words(data, get_tokenizer, allowed_special, mode='train', use_asr_text=False, strip_text=False):
    """ Decode text to chars or BPE
        Inplace operation

        Args:
            data: Iterable[{key, wav, txt, sample_rate}]

        Returns:
            Iterable[{key, wav, txt, tokens, label, sample_rate}]
    """
    tokenizer = get_tokenizer()
    if use_asr_text:
        logging.debug(f"Using asr text.")
    for sample in data:
        if use_asr_text and sample.get('asr_chunks', None) is not None:
            words = [d['word'] for d in sample['asr_chunks']]
            text = ''.join(words)
            sample['asr_text'] = text
        else:
            text = sample['text']
        if strip_text:
            text = text.strip()
        words = text.split(' ')

        text_token = []
        words_begin_index = []
        words_end_index = []
        for i, wrd in enumerate(words):
            if i != 0:
                wrd = ' ' + wrd # NOTE prepend a space to each word for correct tokenization (high probability of aligning with the original one)
            new_subword_token_list = tokenizer.encode(wrd, allowed_special=allowed_special)
            words_begin_index.append(len(text_token))
            text_token.extend(new_subword_token_list)
        if len(words_begin_index) > 0:
            words_end_index = words_begin_index[1:] + [len(text_token)]
        
        sample['text_token'] = text_token
        if mode == 'inference':
            sample['tts_text_token'] = text_token
        # NOTE add words_begin_index
        sample['words_begin_index'] = words_begin_index
        sample['words_end_index'] = words_end_index
        yield sample


def shuffle(data, shuffle_size=10000, mode='train'):
    """ Local shuffle the data

        Args:
            data: Iterable[{key, feat, label}]
            shuffle_size: buffer size for shuffle

        Returns:
            Iterable[{key, feat, label}]
    """
    buf = []
    for sample in data:
        buf.append(sample)
        if len(buf) >= shuffle_size:
            random.shuffle(buf)
            for x in buf:
                yield x
            buf = []
    # The sample left over
    random.shuffle(buf)
    for x in buf:
        yield x


def sort(data, sort_size=500, mode='train'):
    """ Sort the data by feature length.
        Sort is used after shuffle and before batch, so we can group
        utts with similar lengths into a batch, and `sort_size` should
        be less than `shuffle_size`

        Args:
            data: Iterable[{key, feat, label}]
            sort_size: buffer size for sort

        Returns:
            Iterable[{key, feat, label}]
    """

    buf = []
    for sample in data:
        buf.append(sample)
        if len(buf) >= sort_size:
            buf.sort(key=lambda x: x['speech_feat'].size(0))
            for x in buf:
                yield x
            buf = []
    # The sample left over
    buf.sort(key=lambda x: x['speech_feat'].size(0))
    for x in buf:
        yield x


def static_batch(data, batch_size=16):
    """ Static batch the data by `batch_size`

        Args:
            data: Iterable[{key, feat, label}]
            batch_size: batch size

        Returns:
            Iterable[List[{key, feat, label}]]
    """
    buf = []
    for sample in data:
        buf.append(sample)
        if len(buf) >= batch_size:
            yield buf
            buf = []
    if len(buf) > 0:
        yield buf


def dynamic_batch(data, max_frames_in_batch=12000, mode='train'):
    """ Dynamic batch the data until the total frames in batch
        reach `max_frames_in_batch`

        Args:
            data: Iterable[{key, feat, label}]
            max_frames_in_batch: max_frames in one batch

        Returns:
            Iterable[List[{key, feat, label}]]
    """
    buf = []
    longest_frames = 0
    for sample in data:
        assert 'speech_feat' in sample
        assert isinstance(sample['speech_feat'], torch.Tensor)
        new_sample_frames = sample['speech_feat'].size(0)
        longest_frames = max(longest_frames, new_sample_frames)
        frames_after_padding = longest_frames * (len(buf) + 1)
        if frames_after_padding > max_frames_in_batch:
            yield buf
            buf = [sample]
            longest_frames = new_sample_frames
        else:
            buf.append(sample)
    if len(buf) > 0:
        yield buf


def generate_alignment(data, get_tokenizer, allowed_special, mode='train'):
    tokenizer = get_tokenizer()

    for sample in data:
        audio_feature_len = sample['audio_feat'].shape[0]
        asr_token = []
        alignment = []
        for d in sample['asr_chunks']:
            segmented_asr_token = tokenizer.encode(d['word'], allowed_special=allowed_special)
            left, right = [int(x * audio_feature_len) for x in d['range']]
            for _ in range(len(segmented_asr_token)):
                alignment.append([left, right])
            asr_token += segmented_asr_token
        
        sample['asr_text_token'] = asr_token
        sample['asr_alignment'] = alignment
        yield sample


def batch(data, batch_type='static', batch_size=16, max_frames_in_batch=12000, mode='train'):
    """ Wrapper for static/dynamic batch
    """
    if mode == 'inference':
        return static_batch(data, 1)
    else:
        if batch_type == 'static':
            return static_batch(data, batch_size)
        elif batch_type == 'dynamic':
            return dynamic_batch(data, max_frames_in_batch)
        else:
            logging.fatal('Unsupported batch type {}'.format(batch_type))


def padding(data, use_spk_embedding, mode='train', has_audio_branch=False, use_asr_text=True, requires_words_index=False, use_auto_audio_len=True):
    """ Padding the data into training data

        Args:
            data: Iterable[List[{key, feat, label}]]

        Returns:
            Iterable[Tuple(keys, feats, labels, feats lengths, label lengths)]
    """
    for sample in data:
        assert isinstance(sample, list)
        speech_feat_len = torch.tensor([x['speech_feat'].size(1) for x in sample],
                                       dtype=torch.int32)
        order = torch.argsort(speech_feat_len, descending=True)

        utts = [sample[i]['utt'] for i in order]
        speech_token = [torch.tensor(sample[i]['speech_token']) for i in order]
        speech_token_len = torch.tensor([i.size(0) for i in speech_token], dtype=torch.int32)
        speech_token = pad_sequence(speech_token,
                                    batch_first=True,
                                    padding_value=0)
        speech_feat = [sample[i]['speech_feat'] for i in order]
        speech_feat_len = torch.tensor([i.size(0) for i in speech_feat], dtype=torch.int32)
        speech_feat = pad_sequence(speech_feat,
                                   batch_first=True,
                                   padding_value=0)
        text = [sample[i]['text'] for i in order]
        text_token = [torch.tensor(sample[i]['text_token']) for i in order]
        text_token_len = torch.tensor([i.size(0) for i in text_token], dtype=torch.int32)
        text_token = pad_sequence(text_token, batch_first=True, padding_value=0)
        utt_embedding = torch.stack([sample[i]['utt_embedding'] for i in order], dim=0)
        spk_embedding = torch.stack([sample[i]['spk_embedding'] for i in order], dim=0)
        batch = {
            "utts": utts,
            "speech_token": speech_token,
            "speech_token_len": speech_token_len,
            "speech_feat": speech_feat,
            "speech_feat_len": speech_feat_len,
            "text": text,
            "text_token": text_token,
            "text_token_len": text_token_len,
            "utt_embedding": utt_embedding,
            "spk_embedding": spk_embedding,
        }
        if has_audio_branch:
            audio_feat = [sample[i]['audio_feat'] for i in order]
            if use_auto_audio_len:
                audio_feat_len = torch.tensor([i.size(0) for i in audio_feat], dtype=torch.int32)
            else:
                audio_feat_len = torch.tensor([sample[i]['audio_feat_len'] for i in order], dtype=torch.int32)
            audio_feat = pad_sequence(
                audio_feat,
                batch_first=True,
                padding_value=0
            )

            batch.update(
                {
                    "audio_feat": audio_feat,
                    "audio_feat_len": audio_feat_len,
                    "speech_feat": None,
                    "speech_feat_len": None,
                }
            )
            if 'whisper_text_token' in sample[0]:
                whisper_text_token = [torch.tensor(sample[i]['whisper_text_token']) for i in order]
                whisper_text_token_len = torch.tensor([i.size(0) for i in whisper_text_token], dtype=torch.int32)
                whisper_text_token = pad_sequence(whisper_text_token, batch_first=True, padding_value=0)
                batch.update(
                    {
                        "whisper_text_token": whisper_text_token,
                        "whisper_text_token_len": whisper_text_token_len,
                    }
                )
            if use_asr_text and 'asr_text_token' in sample[0]:
                asr_text = [sample[i]['asr_text'] for i in order]
                asr_chunks = [sample[i]['asr_chunks'] for i in order]
                asr_text_token = [torch.tensor(sample[i]['asr_text_token']) for i in order]
                asr_text_token_len = torch.tensor([i.size(0) for i in asr_text_token], dtype=torch.int32)
                asr_text_token = pad_sequence(asr_text_token, batch_first=True, padding_value=0)
                asr_alignment = [torch.tensor(sample[i]['asr_alignment']) for i in order]
                asr_alignment = pad_sequence(asr_alignment, batch_first=True, padding_value=0)

                batch.update(
                    {
                        "asr_text": asr_text,
                        "asr_chunks": asr_chunks,
                        "text_token": asr_text_token,  # Important: replace text_token
                        "text_token_len": asr_text_token_len,  # Important: replace text_token_len
                        "asr_alignment": asr_alignment
                    }
                )
            if requires_words_index:
                words_index = []
                for b, i in enumerate(order):
                    cur_sample = sample[i]
                    for t1, t2 in zip(cur_sample['words_begin_index'], cur_sample['words_end_index']):
                        if t2 - t1 > 1: # >= consider situation only if more than one subword
                            words_index.append((b, t1, t2)) # (b, t1, t2)
                batch.update(
                    {
                        "words_index": words_index,
                    }
                )
        if mode == 'inference':
            tts_text = [sample[i]['tts_text'] for i in order]
            tts_index = [sample[i]['tts_index'] for i in order]
            tts_text_token = [torch.tensor(sample[i]['tts_text_token']) for i in order]
            tts_text_token_len = torch.tensor([i.size(0) for i in tts_text_token], dtype=torch.int32)
            tts_text_token = pad_sequence(tts_text_token, batch_first=True, padding_value=-1)
            batch.update({'tts_text': tts_text,
                          'tts_index': tts_index,
                          'tts_text_token': tts_text_token,
                          'tts_text_token_len': tts_text_token_len})
        if use_spk_embedding is True:
            batch["embedding"] = batch["spk_embedding"]
        else:
            batch["embedding"] = batch["utt_embedding"]
        yield batch

