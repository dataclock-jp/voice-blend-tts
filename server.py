from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response
from starlette.concurrency import run_in_threadpool


BASE_DIR = Path(__file__).resolve().parent
MODEL_NAME = os.getenv("VOICE_BLEND_MODEL", "tts_models/multilingual/multi-dataset/xtts_v2")
VC_MODEL_NAME = os.getenv("VOICE_BLEND_VC_MODEL", "voice_conversion_models/multilingual/vctk/freevc24")
MAX_REFERENCE_FILES = int(os.getenv("VOICE_BLEND_MAX_FILES", "12"))
MAX_WEIGHT_REPETITIONS_PER_FILE = int(os.getenv("VOICE_BLEND_MAX_WEIGHT_REPETITIONS", "8"))
MAX_TOTAL_UPLOAD_MB = int(os.getenv("VOICE_BLEND_MAX_TOTAL_MB", "250"))
MAX_TOTAL_UPLOAD_BYTES = MAX_TOTAL_UPLOAD_MB * 1024 * 1024
SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".aac", ".ogg", ".flac", ".webm"}
LANGUAGES = {"ja", "en", "ko", "zh-cn", "fr", "de", "es", "it", "pt"}
WHISPER_LANGUAGES = {"ja": "ja", "en": "en", "ko": "ko", "zh-cn": "zh", "fr": "fr", "de": "de", "es": "es", "it": "it", "pt": "pt"}
PROFILE_ROOT = Path(os.getenv("VOICE_BLEND_PROFILE_DIR", BASE_DIR / ".runtime" / "vc_profiles"))

app = FastAPI(title="Voice Blend TTS")

_tts_model = None
_vc_model = None
_stt_model = None
_model_lock = threading.RLock()
_vc_lock = threading.RLock()
_stt_lock = threading.RLock()


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (BASE_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    return Response(status_code=204)


@app.get("/api/status")
def status() -> dict[str, str | bool]:
    try:
        import torch  # noqa: F401
        from TTS.api import TTS as _TTS  # noqa: F401
    except Exception as exc:
        return {
            "ready": False,
            "message": f"依存関係が不足しています: {exc.__class__.__name__}",
            "device": "-",
            "model": MODEL_NAME,
        }

    stt_ready = _stt_available()
    return {
        "ready": True,
        "message": "依存関係は利用可能です",
        "device": _device(),
        "model": MODEL_NAME,
        "stt_ready": stt_ready,
        "stt_model": os.getenv("VOICE_BLEND_STT_MODEL", "base"),
        "vc_ready": True,
        "vc_model": VC_MODEL_NAME,
    }


@app.post("/api/transcribe")
async def transcribe(
    language: str = Form("ja"),
    consent_confirmed: bool = Form(False),
    source_audio: UploadFile = File(...),
) -> dict[str, str]:
    if not consent_confirmed:
        raise HTTPException(status_code=400, detail="音声利用の権限と同意を確認してください。")
    if language not in LANGUAGES:
        raise HTTPException(status_code=400, detail=f"未対応の言語です: {language}")
    if not _stt_available():
        raise HTTPException(status_code=501, detail="音声ファイルの文字起こしには openai-whisper が必要です。")

    work_dir = Path(tempfile.mkdtemp(prefix="voice-blend-stt-"))
    try:
        audio_paths = await _save_uploads([source_audio], work_dir)
        text = await run_in_threadpool(_transcribe_file, audio_paths[0], language)
        if not text:
            raise RuntimeError("文字起こし結果が空でした。")
        return {
            "text": text,
            "language": language,
            "model": os.getenv("VOICE_BLEND_STT_MODEL", "base"),
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=_friendly_error(exc)) from exc
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


@app.post("/api/synthesize")
async def synthesize(
    background_tasks: BackgroundTasks,
    text: str = Form(...),
    language: str = Form("ja"),
    split_sentences: bool = Form(True),
    consent_confirmed: bool = Form(False),
    voice_weights: str = Form(""),
    voice_files: list[UploadFile] = File(...),
) -> FileResponse:
    text = text.strip()
    if not consent_confirmed:
        raise HTTPException(status_code=400, detail="音声利用の権限と同意を確認してください。")
    if not text:
        raise HTTPException(status_code=400, detail="生成するセリフを入力してください。")
    if language not in LANGUAGES:
        raise HTTPException(status_code=400, detail=f"未対応の言語です: {language}")
    if not voice_files:
        raise HTTPException(status_code=400, detail="参照音声を1件以上アップロードしてください。")
    if len(voice_files) > MAX_REFERENCE_FILES:
        raise HTTPException(status_code=400, detail=f"参照音声は最大{MAX_REFERENCE_FILES}件までです。")

    work_dir = Path(tempfile.mkdtemp(prefix="voice-blend-"))
    try:
        weights = _parse_voice_weights(voice_weights, len(voice_files))
        reference_paths = await _save_uploads(voice_files, work_dir)
        weighted_reference_paths = _weighted_reference_paths(reference_paths, weights)
        output_name = f"voice-blend-{datetime.now().strftime('%Y%m%d-%H%M%S')}.wav"
        output_path = work_dir / output_name

        await run_in_threadpool(
            _synthesize_to_file,
            text,
            language,
            split_sentences,
            weighted_reference_paths,
            output_path,
        )

        if not output_path.exists() or output_path.stat().st_size == 0:
            raise RuntimeError("モデルは出力ファイルを生成しませんでした。")

        background_tasks.add_task(shutil.rmtree, work_dir, ignore_errors=True)
        return FileResponse(
            output_path,
            media_type="audio/wav",
            filename=output_name,
            headers={"X-Output-Filename": output_name},
            background=background_tasks,
        )
    except HTTPException:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise
    except Exception as exc:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=_friendly_error(exc)) from exc


@app.post("/api/vc/profile")
async def create_vc_profile(
    language: str = Form("ja"),
    split_sentences: bool = Form(True),
    consent_confirmed: bool = Form(False),
    profile_text: str = Form(""),
    voice_weights: str = Form(""),
    voice_files: list[UploadFile] = File(...),
) -> dict[str, str | int]:
    if not consent_confirmed:
        raise HTTPException(status_code=400, detail="音声利用の権限と同意を確認してください。")
    if language not in LANGUAGES:
        raise HTTPException(status_code=400, detail=f"未対応の言語です: {language}")
    if not voice_files:
        raise HTTPException(status_code=400, detail="参照音声を1件以上アップロードしてください。")
    if len(voice_files) > MAX_REFERENCE_FILES:
        raise HTTPException(status_code=400, detail=f"参照音声は最大{MAX_REFERENCE_FILES}件までです。")

    profile_id = uuid.uuid4().hex
    profile_dir = PROFILE_ROOT / profile_id
    profile_dir.mkdir(parents=True, exist_ok=False)

    try:
        weights = _parse_voice_weights(voice_weights, len(voice_files))
        reference_paths = await _save_uploads(voice_files, profile_dir / "refs")
        weighted_reference_paths = _weighted_reference_paths(reference_paths, weights)
        target_path = profile_dir / "target.wav"
        prompt = profile_text.strip() or _default_profile_text(language)
        await run_in_threadpool(
            _synthesize_to_file,
            prompt,
            language,
            split_sentences,
            weighted_reference_paths,
            target_path,
        )
        if not target_path.exists() or target_path.stat().st_size == 0:
            raise RuntimeError("VC用のターゲット参照音声を生成できませんでした。")

        (profile_dir / "meta.txt").write_text(
            f"created_at={int(time.time())}\nlanguage={language}\nmodel={MODEL_NAME}\nvc_model={VC_MODEL_NAME}\n",
            encoding="utf-8",
        )
        return {
            "profile_id": profile_id,
            "language": language,
            "target_seconds": 0,
        }
    except Exception as exc:
        shutil.rmtree(profile_dir, ignore_errors=True)
        if isinstance(exc, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=_friendly_error(exc)) from exc


@app.post("/api/vc/convert")
async def convert_voice(
    background_tasks: BackgroundTasks,
    profile_id: str = Form(...),
    consent_confirmed: bool = Form(False),
    source_audio: UploadFile = File(...),
) -> FileResponse:
    if not consent_confirmed:
        raise HTTPException(status_code=400, detail="音声利用の権限と同意を確認してください。")

    target_path = _profile_target_path(profile_id)
    if not target_path.exists():
        raise HTTPException(status_code=404, detail="VCプロファイルが見つかりません。先にVC準備を実行してください。")

    work_dir = Path(tempfile.mkdtemp(prefix="voice-blend-vc-"))
    try:
        source_paths = await _save_uploads([source_audio], work_dir)
        output_name = f"voice-converted-{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}.wav"
        output_path = work_dir / output_name
        await run_in_threadpool(_voice_convert_to_file, source_paths[0], target_path, output_path)

        if not output_path.exists() or output_path.stat().st_size == 0:
            raise RuntimeError("VCモデルは出力ファイルを生成しませんでした。")

        background_tasks.add_task(shutil.rmtree, work_dir, ignore_errors=True)
        return FileResponse(
            output_path,
            media_type="audio/wav",
            filename=output_name,
            headers={"X-Output-Filename": output_name},
            background=background_tasks,
        )
    except HTTPException:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise
    except Exception as exc:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=_friendly_error(exc)) from exc


async def _save_uploads(files: list[UploadFile], work_dir: Path) -> list[str]:
    work_dir.mkdir(parents=True, exist_ok=True)
    reference_paths: list[str] = []
    total_bytes = 0

    for index, upload in enumerate(files, start=1):
        suffix = Path(upload.filename or "").suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            raise HTTPException(status_code=400, detail=f"未対応の音声形式です: {upload.filename}")

        safe_name = _safe_filename(upload.filename or f"reference-{index}{suffix}")
        target = work_dir / f"{index:02d}-{safe_name}"

        with target.open("wb") as handle:
            while True:
                chunk = await upload.read(1024 * 1024)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > MAX_TOTAL_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"アップロード合計は{MAX_TOTAL_UPLOAD_MB}MBまでです。",
                    )
                handle.write(chunk)

        if target.stat().st_size == 0:
            raise HTTPException(status_code=400, detail=f"空のファイルです: {upload.filename}")
        reference_paths.append(str(target))

    return reference_paths


def _parse_voice_weights(raw: str, expected_count: int) -> list[float]:
    if not raw:
        return [1.0] * expected_count

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="参照音声の重み形式が不正です。") from exc

    if not isinstance(parsed, list) or len(parsed) != expected_count:
        raise HTTPException(status_code=400, detail="参照音声の数と重みの数が一致しません。")

    weights: list[float] = []
    for value in parsed:
        try:
            weight = float(value)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail="参照音声の重みは数値で指定してください。") from exc
        if weight < 0 or weight > 100:
            raise HTTPException(status_code=400, detail="参照音声の重みは0〜100で指定してください。")
        weights.append(weight)

    if not any(weight > 0 for weight in weights):
        raise HTTPException(status_code=400, detail="参照音声の重みは少なくとも1件を0より大きくしてください。")

    return weights


def _weighted_reference_paths(reference_paths: list[str], weights: list[float]) -> list[str]:
    positive = [(path, weight) for path, weight in zip(reference_paths, weights) if weight > 0]
    if not positive:
        raise HTTPException(status_code=400, detail="有効な参照音声の重みがありません。")

    max_weight = max(weight for _, weight in positive)
    min_weight = min(weight for _, weight in positive)
    if abs(max_weight - min_weight) < 1e-6:
        return [path for path, _ in positive]

    weighted_paths: list[str] = []
    for path, weight in positive:
        repeats = max(1, round((weight / max_weight) * MAX_WEIGHT_REPETITIONS_PER_FILE))
        weighted_paths.extend([path] * repeats)
    return weighted_paths


def _synthesize_to_file(
    text: str,
    language: str,
    split_sentences: bool,
    reference_paths: list[str],
    output_path: Path,
) -> None:
    with _model_lock:
        tts = _get_tts_model()
        tts.tts_to_file(
            text=text,
            speaker_wav=reference_paths,
            language=language,
            split_sentences=split_sentences,
            file_path=str(output_path),
        )


def _voice_convert_to_file(source_path: str, target_path: Path, output_path: Path) -> None:
    with _vc_lock:
        vc = _get_vc_model()
        vc.voice_conversion_to_file(
            source_wav=str(source_path),
            target_wav=str(target_path),
            file_path=str(output_path),
        )


def _get_tts_model():
    global _tts_model
    if _tts_model is not None:
        return _tts_model

    from TTS.api import TTS

    model = TTS(MODEL_NAME)
    _tts_model = model.to(_device())
    return _tts_model


def _get_vc_model():
    global _vc_model
    if _vc_model is not None:
        return _vc_model

    from TTS.api import TTS

    model = TTS(model_name=VC_MODEL_NAME, progress_bar=False)
    _vc_model = model.to(_device())
    return _vc_model


def _transcribe_file(audio_path: str, language: str) -> str:
    with _stt_lock:
        model = _get_stt_model()
        kwargs = {
            "fp16": _device() == "cuda",
            "verbose": False,
        }
        whisper_language = WHISPER_LANGUAGES.get(language)
        if whisper_language:
            kwargs["language"] = whisper_language
        result = model.transcribe(audio_path, **kwargs)
    return str(result.get("text", "")).strip()


def _get_stt_model():
    global _stt_model
    if _stt_model is not None:
        return _stt_model

    import whisper

    model_name = os.getenv("VOICE_BLEND_STT_MODEL", "base")
    _stt_model = whisper.load_model(model_name, device=_device())
    return _stt_model


def _stt_available() -> bool:
    try:
        import whisper  # noqa: F401
    except Exception:
        return False
    return True


def _device() -> str:
    if os.getenv("VOICE_BLEND_FORCE_CPU", "").lower() in {"1", "true", "yes"}:
        return "cpu"

    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _safe_filename(name: str) -> str:
    suffix = Path(name).suffix.lower()
    stem = Path(name).stem or "reference"
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-") or "reference"
    return f"{safe_stem}{suffix}"


def _profile_target_path(profile_id: str) -> Path:
    safe_id = re.sub(r"[^A-Za-z0-9_-]+", "", profile_id)
    if not safe_id or safe_id != profile_id:
        raise HTTPException(status_code=400, detail="VCプロファイルIDが不正です。")
    return PROFILE_ROOT / safe_id / "target.wav"


def _default_profile_text(language: str) -> str:
    if language == "ja":
        return "これはリアルタイム音声変換のための基準音声です。自然な速さで、はっきりと話しています。"
    if language == "en":
        return "This is a reference voice for real time voice conversion, spoken clearly at a natural pace."
    return "This is a reference voice for real time voice conversion."


def _friendly_error(exc: Exception) -> str:
    text = str(exc)
    if "No module named" in text or exc.__class__.__name__ == "ModuleNotFoundError":
        return "Python依存関係が不足しています。READMEのインストール手順を実行してください。"
    if "CUDA out of memory" in text:
        return "GPUメモリが不足しています。VOICE_BLEND_FORCE_CPU=1でCPU実行に切り替えるか、短いセリフで再試行してください。"
    if "ffmpeg" in text.lower():
        return "音声デコードに失敗しました。ffmpegをインストールするか、WAV形式の参照音声を使ってください。"
    if "Failed to load audio" in text or "No such file or directory" in text:
        return "音声ファイルの読み込みに失敗しました。ffmpegをインストールするか、WAV形式で再試行してください。"
    if "not enough values to unpack" in text or "Input signal length" in text:
        return "入力音声が短すぎます。2〜5秒以上の発話チャンクで再試行してください。"
    return text or "音声生成中に不明なエラーが発生しました。"
