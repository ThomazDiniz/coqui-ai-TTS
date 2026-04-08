import gc
import logging
import os
import time

import librosa
import pandas
import soundfile as sf
import torch
from faster_whisper import WhisperModel
from tqdm import tqdm

# torch.set_num_threads(1)
from TTS.tts.layers.xtts.tokenizer import multilingual_cleaners

torch.set_num_threads(16)

_LOG = logging.getLogger("xtts_ft.formatter")

audio_types = (".wav", ".mp3", ".flac")


def _load_audio(path: str):
    """Load [channels, samples] float32; avoids torchaudio/torchcodec (needs full CUDA NPP in Docker base images)."""
    y, sr = librosa.load(path, sr=None, mono=False)
    if y.ndim == 1:
        wav = torch.from_numpy(y.astype("float32", copy=False)).unsqueeze(0)
    else:
        wav = torch.from_numpy(y.astype("float32", copy=False))
    return wav, int(sr)


def list_audios(basePath, contains=None):
    # return the set of files that are valid
    return list_files(basePath, validExts=audio_types, contains=contains)


def list_files(basePath, validExts=None, contains=None):
    # loop over the directory structure
    for rootDir, dirNames, filenames in os.walk(basePath):
        # loop over the filenames in the current directory
        for filename in filenames:
            # if the contains string is not none and the filename does not contain
            # the supplied string, then ignore the file
            if contains is not None and filename.find(contains) == -1:
                continue

            # determine the file extension of the current file
            ext = filename[filename.rfind(".") :].lower()

            # check to see if the file is an audio and should be processed
            if validExts is None or ext.endswith(validExts):
                # construct the path to the audio and yield it
                audioPath = os.path.join(rootDir, filename)
                yield audioPath


def format_audio_list(
    audio_files,
    target_language="en",
    out_path=None,
    buffer=0.2,
    eval_percentage=0.15,
    speaker_name="coqui",
    gradio_progress=None,
):
    if isinstance(audio_files, str):
        audio_files = [audio_files]
    n_in = len(audio_files) if audio_files is not None else 0
    _LOG.info(
        "[DATASET] format_audio_list begin files=%s lang=%s out=%s eval%%=%s buffer=%s speaker=%s",
        n_in,
        target_language,
        out_path,
        eval_percentage * 100.0,
        buffer,
        speaker_name,
    )
    audio_total_size = 0
    # make sure that ooutput file exists
    os.makedirs(out_path, exist_ok=True)

    # Loading Whisper
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _LOG.info("[DATASET] whisper device=%s cuda_available=%s", device, torch.cuda.is_available())

    print("Loading Whisper Model!")
    t0 = time.perf_counter()
    asr_model = WhisperModel("large-v2", device=device, compute_type="float16")
    _LOG.info("[DATASET] WhisperModel loaded in %.1fs", time.perf_counter() - t0)

    metadata = {"audio_file": [], "text": [], "speaker_name": []}

    if gradio_progress is not None:
        tqdm_object = gradio_progress.tqdm(audio_files, desc="Formatting...")
    else:
        tqdm_object = tqdm(audio_files)

    for fi, audio_path in enumerate(tqdm_object):
        if fi % 25 == 0:
            _LOG.info("[DATASET] processing file %s/%s: %s", fi + 1, n_in, audio_path)
        wav, sr = _load_audio(audio_path)
        # stereo to mono if needed
        if wav.size(0) != 1:
            wav = torch.mean(wav, dim=0, keepdim=True)

        wav = wav.squeeze()
        dur_s = wav.size(-1) / sr
        audio_total_size += dur_s

        t_tr = time.perf_counter()
        segments, _ = asr_model.transcribe(audio_path, word_timestamps=True, language=target_language)
        if fi % 25 == 0:
            _LOG.info("[DATASET] transcribe %.2fs audio in %.2fs", dur_s, time.perf_counter() - t_tr)
        segments = list(segments)
        i = 0
        sentence = ""
        sentence_start = None
        first_word = True
        # added all segments words in a unique list
        words_list = []
        for _, segment in enumerate(segments):
            words = list(segment.words)
            words_list.extend(words)

        # process each word
        for word_idx, word in enumerate(words_list):
            if first_word:
                sentence_start = word.start
                # If it is the first sentence, add buffer or get the begining of the file
                if word_idx == 0:
                    sentence_start = max(sentence_start - buffer, 0)  # Add buffer to the sentence start
                else:
                    # get previous sentence end
                    previous_word_end = words_list[word_idx - 1].end
                    # add buffer or get the silence midle between the previous sentence and the current one
                    sentence_start = max(sentence_start - buffer, (previous_word_end + sentence_start) / 2)

                sentence = word.word
                first_word = False
            else:
                sentence += word.word

            if word.word[-1] in ["!", ".", "?"]:
                sentence = sentence[1:]
                # Expand number and abbreviations plus normalization
                sentence = multilingual_cleaners(sentence, target_language)
                audio_file_name, _ = os.path.splitext(os.path.basename(audio_path))

                audio_file = f"wavs/{audio_file_name}_{str(i).zfill(8)}.wav"

                # Check for the next word's existence
                if word_idx + 1 < len(words_list):
                    next_word_start = words_list[word_idx + 1].start
                else:
                    # If don't have more words it means that it is the last sentence then use the audio len as next word start
                    next_word_start = (wav.shape[0] - 1) / sr

                # Average the current word end and next word start
                word_end = min((word.end + next_word_start) / 2, word.end + buffer)

                absoulte_path = os.path.join(out_path, audio_file)
                os.makedirs(os.path.dirname(absoulte_path), exist_ok=True)
                i += 1
                first_word = True

                audio = wav[int(sr * sentence_start) : int(sr * word_end)].unsqueeze(0)
                # if the audio is too short ignore it (i.e < 0.33 seconds)
                if audio.size(-1) >= sr / 3:
                    sf.write(absoulte_path, audio.squeeze(0).cpu().numpy(), sr, subtype="PCM_16")
                else:
                    continue

                metadata["audio_file"].append(audio_file)
                metadata["text"].append(sentence)
                metadata["speaker_name"].append(speaker_name)

    df = pandas.DataFrame(metadata)
    _LOG.info("[DATASET] extracted segments rows=%s total_audio_s=%.2f", len(df), audio_total_size)
    df = df.sample(frac=1)
    num_val_samples = int(len(df) * eval_percentage)

    df_eval = df[:num_val_samples]
    df_train = df[num_val_samples:]

    df_train = df_train.sort_values("audio_file")
    train_metadata_path = os.path.join(out_path, "metadata_train.csv")
    df_train.to_csv(train_metadata_path, sep="|", index=False)

    eval_metadata_path = os.path.join(out_path, "metadata_eval.csv")
    df_eval = df_eval.sort_values("audio_file")
    df_eval.to_csv(eval_metadata_path, sep="|", index=False)
    _LOG.info(
        "[DATASET] wrote train=%s rows=%s eval=%s rows=%s",
        train_metadata_path,
        len(df_train),
        eval_metadata_path,
        len(df_eval),
    )

    # deallocate VRAM and RAM
    del asr_model, df_train, df_eval, df, metadata
    gc.collect()

    _LOG.info("[DATASET] format_audio_list done total_audio_s=%.2f", audio_total_size)
    return train_metadata_path, eval_metadata_path, audio_total_size
