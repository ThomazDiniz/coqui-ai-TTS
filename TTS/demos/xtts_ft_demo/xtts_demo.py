import argparse
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from typing import Final

import gradio as gr
import torch
import torchaudio

from TTS.demos.xtts_ft_demo.utils.formatter import format_audio_list
from TTS.demos.xtts_ft_demo.utils.gpt_train import train_gpt
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

# Shared by dataset (Tab 1), training (Tab 2), and inference (Tab 3) language dropdowns.
XTTS_FT_LANG_CHOICES = [
    "en",
    "es",
    "fr",
    "de",
    "it",
    "pt",
    "pl",
    "tr",
    "ru",
    "nl",
    "cs",
    "ar",
    "zh",
    "hu",
    "ko",
    "ja",
    "hi",
]


def clear_gpu_cache():
    # clear the GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


XTTS_MODEL = None


def load_model(xtts_checkpoint, xtts_config, xtts_vocab):
    global XTTS_MODEL
    UI_LOG.info("[STEP][LOAD_MODEL] begin ckpt=%s cfg=%s vocab=%s", xtts_checkpoint, xtts_config, xtts_vocab)
    clear_gpu_cache()
    if not xtts_checkpoint or not xtts_config or not xtts_vocab:
        UI_LOG.warning("[STEP][LOAD_MODEL] missing path(s); abort")
        return "You need to run the previous steps or manually set the `XTTS checkpoint path`, `XTTS config path`, and `XTTS vocab path` fields !!"
    try:
        for label, p in (("ckpt", xtts_checkpoint), ("cfg", xtts_config), ("vocab", xtts_vocab)):
            try:
                exists = os.path.isfile(p)
                size_mb = (os.path.getsize(p) / 1024**2) if exists else None
                UI_LOG.info("[STEP][LOAD_MODEL] %s exists=%s size_mb=%s path=%s", label, exists, size_mb, p)
            except Exception:
                UI_LOG.exception("[STEP][LOAD_MODEL] stat failed for %s=%s", label, p)

        t0 = time.perf_counter()
        UI_LOG.info("[STEP][LOAD_MODEL] load_json begin")
        config = XttsConfig()
        config.load_json(xtts_config)
        UI_LOG.info("[STEP][LOAD_MODEL] load_json done in %.2fs", time.perf_counter() - t0)

        t1 = time.perf_counter()
        UI_LOG.info("[STEP][LOAD_MODEL] init_from_config begin")
        XTTS_MODEL = Xtts.init_from_config(config)
        UI_LOG.info("[STEP][LOAD_MODEL] init_from_config done in %.2fs", time.perf_counter() - t1)

        t2 = time.perf_counter()
        UI_LOG.info("[STEP][LOAD_MODEL] load_checkpoint begin use_deepspeed=False")
        print("Loading XTTS model! ")
        XTTS_MODEL.load_checkpoint(config, checkpoint_path=xtts_checkpoint, vocab_path=xtts_vocab, use_deepspeed=False)
        UI_LOG.info("[STEP][LOAD_MODEL] load_checkpoint done in %.2fs", time.perf_counter() - t2)

        if torch.cuda.is_available():
            t3 = time.perf_counter()
            UI_LOG.info("[STEP][LOAD_MODEL] cuda() begin")
            XTTS_MODEL.cuda()
            UI_LOG.info("[STEP][LOAD_MODEL] cuda() done in %.2fs", time.perf_counter() - t3)
            UI_LOG.info(
                "[STEP][LOAD_MODEL] cuda allocated_mb=%.1f reserved_mb=%.1f",
                torch.cuda.memory_allocated() / 1024**2,
                torch.cuda.memory_reserved() / 1024**2,
            )

        print("Model Loaded!")
        UI_LOG.info("[STEP][LOAD_MODEL] done")
        return "Model Loaded!"
    except Exception:
        UI_LOG.exception("[STEP][LOAD_MODEL] failed")
        raise


def run_tts(lang, tts_text, speaker_audio_file):
    UI_LOG.info("[STEP][INFERENCE] begin lang=%s text_len=%s ref=%s", lang, len(tts_text or ""), speaker_audio_file)
    if XTTS_MODEL is None or not speaker_audio_file:
        UI_LOG.warning("[STEP][INFERENCE] model not loaded or no reference audio")
        return "You need to run the previous step to load the model !!", None, None

    try:
        if torch.cuda.is_available():
            UI_LOG.info(
                "[STEP][INFERENCE] cuda allocated_mb=%.1f reserved_mb=%.1f",
                torch.cuda.memory_allocated() / 1024**2,
                torch.cuda.memory_reserved() / 1024**2,
            )

        t0 = time.perf_counter()
        UI_LOG.info(
            "[STEP][INFERENCE] conditioning begin gpt_cond_len=%s max_ref_len=%s sound_norm_refs=%s",
            getattr(XTTS_MODEL.config, "gpt_cond_len", None),
            getattr(XTTS_MODEL.config, "max_ref_len", None),
            getattr(XTTS_MODEL.config, "sound_norm_refs", None),
        )
        gpt_cond_latent, speaker_embedding = XTTS_MODEL.get_conditioning_latents(
            audio_path=speaker_audio_file,
            gpt_cond_len=XTTS_MODEL.config.gpt_cond_len,
            max_ref_length=XTTS_MODEL.config.max_ref_len,
            sound_norm_refs=XTTS_MODEL.config.sound_norm_refs,
        )
        UI_LOG.info("[STEP][INFERENCE] conditioning done in %.2fs", time.perf_counter() - t0)
        if torch.cuda.is_available():
            UI_LOG.info(
                "[STEP][INFERENCE] cuda(after conditioning) allocated_mb=%.1f reserved_mb=%.1f",
                torch.cuda.memory_allocated() / 1024**2,
                torch.cuda.memory_reserved() / 1024**2,
            )

        t1 = time.perf_counter()
        UI_LOG.info("[STEP][INFERENCE] inference begin")
        out = XTTS_MODEL.inference(
            text=tts_text,
            language=lang,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            temperature=XTTS_MODEL.config.temperature,  # Add custom parameters here
            length_penalty=XTTS_MODEL.config.length_penalty,
            repetition_penalty=XTTS_MODEL.config.repetition_penalty,
            top_k=XTTS_MODEL.config.top_k,
            top_p=XTTS_MODEL.config.top_p,
        )
        UI_LOG.info("[STEP][INFERENCE] inference done in %.2fs", time.perf_counter() - t1)
        if torch.cuda.is_available():
            UI_LOG.info(
                "[STEP][INFERENCE] cuda(after inference) allocated_mb=%.1f reserved_mb=%.1f",
                torch.cuda.memory_allocated() / 1024**2,
                torch.cuda.memory_reserved() / 1024**2,
            )
    except Exception:
        UI_LOG.exception("[STEP][INFERENCE] failed")
        raise

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fp:
        out["wav"] = torch.tensor(out["wav"]).unsqueeze(0)
        out_path = fp.name
        torchaudio.save(out_path, out["wav"], 24000)

    UI_LOG.info("[STEP][INFERENCE] done out=%s", out_path)
    return "Speech generated !", out_path, speaker_audio_file


# define a logger to redirect
_ORIGINAL_STDOUT = sys.stdout


class Logger:
    def __init__(self, filename: str | None = None):
        # Same stream as Gradio "Logs" + optional path on a host volume (e.g. XTTS_FT_LOG_FILE=/data/xtts_ft/ui_console.log).
        self.log_file = filename or os.environ.get("XTTS_FT_LOG_FILE", "log.out")
        _abs = os.path.abspath(self.log_file)
        _parent = os.path.dirname(_abs)
        if _parent:
            os.makedirs(_parent, exist_ok=True)
        # Always mirror to the process real stdout (e.g. Docker logs), not only the Gradio tail file.
        self.terminal = _ORIGINAL_STDOUT
        self.log = open(self.log_file, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def isatty(self):
        return False


# redirect stdout and stderr to a file
sys.stdout = Logger()
sys.stderr = sys.stdout


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

UI_LOG = logging.getLogger("xtts_ft.ui")
UI_LOG.setLevel(logging.INFO)

_LOG_POLL: dict[str, int | float] = {"n": 0, "slow_warned": 0}


# Smaller tail = faster reads on slow volumes while training writes heavily to log.out
_LOG_TAIL_BYTES: Final[int] = 32 * 1024

# Log accordion refresh interval (seconds). Lower = fresher logs, more I/O load on the Gradio process.
def _log_poll_interval_sec() -> int:
    try:
        return max(3, int(os.environ.get("XTTS_FT_LOG_POLL_SEC", "10")))
    except ValueError:
        return 10


_LOG_POLL_INTERVAL_SEC: Final[int] = _log_poll_interval_sec()


def read_logs():
    """Tail log file for Gradio; logs slow reads to spot UI stalls."""
    t0 = time.perf_counter()
    sys.stdout.flush()
    file_size = 0
    try:
        with open(sys.stdout.log_file, "rb") as f:
            f.seek(0, os.SEEK_END)
            file_size = f.tell()
            f.seek(max(0, file_size - _LOG_TAIL_BYTES), os.SEEK_SET)
            data = f.read()
        text = data.decode("utf-8", errors="replace")
    except Exception:
        UI_LOG.exception("[UI][LOG_POLL] failed to read log tail")
        return ""
    dt_ms = (time.perf_counter() - t0) * 1000.0
    _LOG_POLL["n"] = int(_LOG_POLL["n"]) + 1
    n = int(_LOG_POLL["n"])
    if dt_ms > 500 or n % 25 == 1:
        UI_LOG.info(
            "[UI][LOG_POLL] call=%s duration_ms=%.1f tail_bytes=%s log_file_size=%s",
            n,
            dt_ms,
            len(data),
            file_size,
        )
    if dt_ms > 500 and not _LOG_POLL.get("slow_warned"):
        UI_LOG.warning(
            "[UI][LOG_POLL] slow read (>500ms) — if the browser freezes, increase Timer interval or disable log polling."
        )
        _LOG_POLL["slow_warned"] = 1
    return text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""XTTS fine-tuning demo\n\n"""
        """
        Example runs:
        python3 TTS/demos/xtts_ft_demo/xtts_demo.py --port
        """,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--port",
        type=int,
        help="Port to run the gradio demo. Default: 5003",
        default=5003,
    )
    parser.add_argument(
        "--out_path",
        type=str,
        help="Output path (where data and checkpoints will be saved) Default: /tmp/xtts_ft/",
        default="/tmp/xtts_ft/",
    )

    parser.add_argument(
        "--num_epochs",
        type=int,
        help="Number of epochs to train. Default: 10",
        default=10,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size (lower uses less VRAM; Docker OOM/exit 137 often fixed by reducing this). Default: 2",
        default=2,
    )
    parser.add_argument(
        "--grad_acumm",
        type=int,
        help="Grad accumulation steps. Default: 2",
        default=2,
    )
    parser.add_argument(
        "--max_audio_length",
        type=int,
        help="Max permitted audio size in seconds. Default: 11",
        default=11,
    )
    parser.add_argument(
        "--share",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Create a temporary public gradio.app link (off by default; use for Colab/remote access only).",
    )
    parser.add_argument(
        "--analytics",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow Gradio analytics/telemetry (default: off for local/Docker).",
    )

    args = parser.parse_args()
    UI_LOG.info(
        "[BOOT] pid=%s cwd=%s torch=%s cuda=%s gradio=%s",
        os.getpid(),
        os.getcwd(),
        getattr(torch, "__version__", "?"),
        torch.cuda.is_available(),
        gr.__version__,
    )
    UI_LOG.info(
        "[BOOT] args port=%s out_path=%s num_epochs=%s batch_size=%s grad_acumm=%s max_audio_length=%s share=%s analytics=%s",
        args.port,
        args.out_path,
        args.num_epochs,
        args.batch_size,
        args.grad_acumm,
        args.max_audio_length,
        args.share,
        args.analytics,
    )
    UI_LOG.info(
        "[BOOT] log poll every %ss (XTTS_FT_LOG_POLL_SEC); exit code 137 usually means OOM — raise Docker memory or lower batch size",
        _LOG_POLL_INTERVAL_SEC,
    )

    with gr.Blocks(analytics_enabled=args.analytics) as demo:
        with gr.Tab("1 - Data processing") as tab_data:
            out_path = gr.Textbox(
                label="Output path (where data and checkpoints will be saved):",
                value=args.out_path,
            )
            upload_file = gr.File(
                file_count="multiple",
                label="Select here the audio files that you want to use for XTTS trainining (Supported formats: wav, mp3, and flac)",
            )
            lang = gr.Dropdown(
                label="Dataset Language",
                value="pt",
                choices=XTTS_FT_LANG_CHOICES,
            )
            progress_data = gr.Label(label="Progress:")
            prompt_compute_btn = gr.Button(value="Step 1 - Create dataset")

            def preprocess_dataset(audio_path, language, out_path, progress=gr.Progress(track_tqdm=True)):
                n_files = len(audio_path) if audio_path else 0
                UI_LOG.info("[STEP][DATASET] begin language=%s base_out=%s uploaded_files=%s", language, out_path, n_files)
                clear_gpu_cache()
                out_path = os.path.join(out_path, "dataset")
                os.makedirs(out_path, exist_ok=True)
                if audio_path is None:
                    UI_LOG.warning("[STEP][DATASET] aborted: no audio_path (upload not ready?)")
                    return (
                        "You should provide one or multiple audio files! If you provided it, probably the upload of the files is not finished yet!",
                        "",
                        "",
                        gr.update(),
                    )
                else:
                    try:
                        UI_LOG.info("[STEP][DATASET] calling format_audio_list out_path=%s", out_path)
                        train_meta, eval_meta, audio_total_size = format_audio_list(
                            audio_path, target_language=language, out_path=out_path, gradio_progress=progress
                        )
                        UI_LOG.info("[STEP][DATASET] format_audio_list ok audio_total_s=%.2f", audio_total_size)
                    except:
                        UI_LOG.exception("[STEP][DATASET] format_audio_list failed")
                        traceback.print_exc()
                        error = traceback.format_exc()
                        return (
                            f"The data processing was interrupted due an error !! Please check the console to verify the full error message! \n Error summary: {error}",
                            "",
                            "",
                            gr.update(),
                        )

                clear_gpu_cache()

                # if audio total len is less than 2 minutes raise an error
                if audio_total_size < 120:
                    message = "The sum of the duration of the audios that you provided should be at least 2 minutes!"
                    print(message)
                    UI_LOG.warning("[STEP][DATASET] too short: %.2fs (<120s)", audio_total_size)
                    return message, "", "", gr.update()

                print("Dataset Processed!")
                UI_LOG.info("[STEP][DATASET] done train_meta=%s eval_meta=%s", train_meta, eval_meta)
                return "Dataset Processed!", train_meta, eval_meta, gr.update(value=language)

        with gr.Tab("2 - Fine-tuning XTTS Encoder") as tab_ft:
            train_csv = gr.Textbox(
                label="Train CSV:",
            )
            eval_csv = gr.Textbox(
                label="Eval CSV:",
            )
            train_lang = gr.Dropdown(
                label="Training language (must match dataset text language from Step 1)",
                value="pt",
                choices=XTTS_FT_LANG_CHOICES,
            )
            num_epochs = gr.Slider(
                label="Number of epochs:",
                minimum=1,
                maximum=100,
                step=1,
                value=args.num_epochs,
            )
            batch_size = gr.Slider(
                label="Batch size:",
                minimum=2,
                maximum=512,
                step=1,
                value=args.batch_size,
            )
            grad_acumm = gr.Slider(
                label="Grad accumulation steps:",
                minimum=1,
                maximum=128,
                step=1,
                value=args.grad_acumm,
            )
            max_audio_length = gr.Slider(
                label="Max permitted audio size in seconds:",
                minimum=2,
                maximum=20,
                step=1,
                value=args.max_audio_length,
            )
            progress_train = gr.Label(label="Progress:")
            loss_plot_img = gr.Image(
                label="Training loss (also saved as training_loss.png in the run folder)",
            )
            train_btn = gr.Button(value="Step 2 - Run the training")

            def train_model(
                language, train_csv, eval_csv, num_epochs, batch_size, grad_acumm, output_path, max_audio_length
            ):
                UI_LOG.info(
                    "[STEP][TRAIN] begin language=%s epochs=%s batch=%s grad_acumm=%s out=%s max_audio_s=%s train_csv=%s eval_csv=%s",
                    language,
                    num_epochs,
                    batch_size,
                    grad_acumm,
                    output_path,
                    max_audio_length,
                    train_csv,
                    eval_csv,
                )
                if torch.cuda.is_available():
                    UI_LOG.info("[STEP][TRAIN] cuda_mem_allocated_mb=%.1f", torch.cuda.memory_allocated() / 1024**2)
                clear_gpu_cache()
                if not train_csv or not eval_csv:
                    UI_LOG.warning("[STEP][TRAIN] aborted: missing train_csv or eval_csv")
                    return (
                        "You need to run the data processing step or manually set `Train CSV` and `Eval CSV` fields !",
                        "",
                        "",
                        "",
                        "",
                        None,
                    )
                try:
                    # convert seconds to waveform frames
                    max_audio_length = int(max_audio_length * 22050)
                    UI_LOG.info("[STEP][TRAIN] calling train_gpt max_audio_length_frames=%s", max_audio_length)
                    t_train0 = time.perf_counter()
                    (
                        config_path,
                        original_xtts_checkpoint,
                        vocab_file,
                        exp_path,
                        speaker_wav,
                        loss_plot_path,
                    ) = train_gpt(
                        language,
                        num_epochs,
                        batch_size,
                        grad_acumm,
                        train_csv,
                        eval_csv,
                        output_path=output_path,
                        max_audio_length=max_audio_length,
                    )
                    UI_LOG.info(
                        "[STEP][TRAIN] train_gpt finished in %.1fs exp_path=%s loss_plot=%s",
                        time.perf_counter() - t_train0,
                        exp_path,
                        loss_plot_path,
                    )
                except:
                    UI_LOG.exception("[STEP][TRAIN] train_gpt raised")
                    traceback.print_exc()
                    error = traceback.format_exc()
                    return (
                        f"The training was interrupted due an error !! Please check the console to check the full error message! \n Error summary: {error}",
                        "",
                        "",
                        "",
                        "",
                        None,
                    )

                # Copy original files to the run folder (Windows-safe; avoids relying on `cp`).
                copied_config_path = config_path
                copied_vocab_path = vocab_file
                for label, src in (("config", config_path), ("vocab", vocab_file)):
                    try:
                        dst = os.path.join(exp_path, os.path.basename(src))
                        shutil.copy2(src, dst)
                        UI_LOG.info("[STEP][TRAIN] copied %s %s -> %s", label, src, dst)
                        if label == "config":
                            copied_config_path = dst
                        elif label == "vocab":
                            copied_vocab_path = dst
                    except Exception:
                        UI_LOG.exception("[STEP][TRAIN] failed to copy %s %s into %s", label, src, exp_path)

                ft_xtts_checkpoint = os.path.join(exp_path, "best_model.pth")
                print("Model training done!")
                clear_gpu_cache()
                plot_path = loss_plot_path if loss_plot_path and os.path.isfile(loss_plot_path) else None
                if torch.cuda.is_available():
                    UI_LOG.info("[STEP][TRAIN] cuda_mem_allocated_mb=%.1f (after)", torch.cuda.memory_allocated() / 1024**2)
                UI_LOG.info("[STEP][TRAIN] done checkpoint=%s plot=%s", ft_xtts_checkpoint, plot_path)
                return "Model training done!", copied_config_path, copied_vocab_path, ft_xtts_checkpoint, speaker_wav, plot_path

        with gr.Tab("3 - Inference") as tab_inf:
            with gr.Row():
                with gr.Column() as col1:
                    xtts_checkpoint = gr.Textbox(
                        label="XTTS checkpoint path:",
                        value="",
                    )
                    xtts_config = gr.Textbox(
                        label="XTTS config path:",
                        value="",
                    )

                    xtts_vocab = gr.Textbox(
                        label="XTTS vocab path:",
                        value="",
                    )
                    progress_load = gr.Label(label="Progress:")
                    load_btn = gr.Button(value="Step 3 - Load Fine-tuned XTTS model")

                with gr.Column() as col2:
                    speaker_reference_audio = gr.Textbox(
                        label="Speaker reference audio:",
                        value=r"E:\git\coqui-ai-TTS\PB_0001.wav",
                    )
                    tts_language = gr.Dropdown(
                        label="Language",
                        value="pt",
                        choices=XTTS_FT_LANG_CHOICES,
                    )
                    tts_text = gr.Textbox(
                        label="Input Text.",
                        value="O céu de Campina Grande ficou alaranjado no fim da tarde.",
                    )
                    tts_btn = gr.Button(value="Step 4 - Inference")

                with gr.Column() as col3:
                    progress_gen = gr.Label(label="Progress:")
                    tts_output_audio = gr.Audio(label="Generated Audio.")
                    reference_audio = gr.Audio(label="Reference audio used.")

        # Single log viewer. With queue(default_concurrency_limit=1), Timer+Textbox(every=...) queues each
        # tick and blocks tab clicks — use tick(..., queue=False) so polling does not serialize the UI.
        with gr.Accordion("Logs (terminal output)", open=False):
            log_timer = gr.Timer(_LOG_POLL_INTERVAL_SEC)
            shared_logs = gr.Textbox(
                label="Logs:",
                interactive=False,
                lines=18,
                max_lines=40,
                value="",
            )
            log_timer.tick(
                read_logs,
                inputs=None,
                outputs=shared_logs,
                queue=False,
                show_progress="hidden",
            )
            demo.load(read_logs, outputs=shared_logs, queue=False)

        prompt_compute_btn.click(
            fn=preprocess_dataset,
            inputs=[
                upload_file,
                lang,
                out_path,
            ],
            outputs=[
                progress_data,
                train_csv,
                eval_csv,
                train_lang,
            ],
        )

        train_btn.click(
            fn=train_model,
            inputs=[
                train_lang,
                train_csv,
                eval_csv,
                num_epochs,
                batch_size,
                grad_acumm,
                out_path,
                max_audio_length,
            ],
            outputs=[progress_train, xtts_config, xtts_vocab, xtts_checkpoint, speaker_reference_audio, loss_plot_img],
        )

        load_btn.click(
            fn=load_model,
            inputs=[xtts_checkpoint, xtts_config, xtts_vocab],
            outputs=[progress_load],
        )

        tts_btn.click(
            fn=run_tts,
            inputs=[
                tts_language,
                tts_text,
                speaker_reference_audio,
            ],
            outputs=[progress_gen, tts_output_audio, reference_audio],
        )

        def _log_tab_select(evt: gr.SelectData):
            UI_LOG.info("[UI][TAB] selected index=%r value=%r selected=%s", evt.index, evt.value, evt.selected)

        tab_data.select(_log_tab_select, queue=False)
        tab_ft.select(_log_tab_select, queue=False)
        tab_inf.select(_log_tab_select, queue=False)

        UI_LOG.info(
            "[UI] handlers registered: step1(dataset), step2(train), step3(load), step4(inference), tabs=%s",
            3,
        )

        def _on_app_load():
            UI_LOG.info(
                "[UI][LOAD] demo.load fired (session) gradio=%s torch_cuda=%s pid=%s",
                gr.__version__,
                torch.cuda.is_available(),
                os.getpid(),
            )

        demo.load(_on_app_load)

    demo.queue(default_concurrency_limit=1)
    UI_LOG.info("[UI][LAUNCH] server_name=0.0.0.0 port=%s queue_concurrency=1 share=%s", args.port, args.share)
    demo.launch(share=args.share, debug=False, server_port=args.port, server_name="0.0.0.0")
