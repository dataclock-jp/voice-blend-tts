# Voice Blend TTS

複数の参照音声から話者条件を抽出し、その条件を使って入力テキストを読み上げる、または入力音声をボイスチェンジするローカルツールです。

## 仕組み

- フロントエンドで参照音声、言語、セリフを指定します。
- FastAPIバックエンドがアップロード音声を一時ファイルに保存します。
- Coqui XTTS v2に複数の `speaker_wav` を渡し、モデル側で声質の条件特徴を抽出して合成します。
- 生成したWAVをブラウザでプレビュー、ダウンロードできます。
- 置換元音声をアップロードした場合はWhisperで文字起こしし、そのテキストを同じ合成声で読み上げます。
- マイク入力はブラウザの音声認識で確定発話を取り込み、必要に応じて発話単位で逐次生成します。
- 本格VCモードでは、参照音声群からXTTSで混合声のターゲット参照WAVを作り、Coqui FreeVC24で置換元音声を音声対音声変換します。

これは単純な波形ミキシングではありません。セリフやタイミングが一致しない参照音声でも使えますが、複数人の声を完全に平均化した声になるとは限りません。

## セットアップ

```powershell
git clone https://github.com/dataclock-jp/voice-blend-tts.git
cd voice-blend-tts
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# XTTS v2のモデルライセンスを確認して同意した場合だけ設定します。
$env:COQUI_TOS_AGREED = "1"
uvicorn server:app --reload --host 127.0.0.1 --port 8000
```

ブラウザで `http://127.0.0.1:8000` を開きます。

初回生成時にXTTS v2のモデルがダウンロードされます。GPUがある場合はCUDA、ない場合はCPUで実行します。Python 3.14ではCoqui TTSの依存関係が合わない可能性が高いため、Python 3.10を使ってください。

## GPU / CUDA

NVIDIA GPUで実行する場合は、CPU版PyTorchではなくCUDA版PyTorchを入れてください。RTX 3070と新しめのNVIDIAドライバではCUDA 12.8 wheelを推奨します。

```powershell
.\.venv\Scripts\Activate.ps1
pip uninstall -y torch torchaudio torchvision
pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu128
```

確認:

```powershell
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

`torch.cuda.is_available()` が `True` なら、画面のステータスは `準備完了 (cuda)` になります。

## Troubleshooting

`cannot import name 'BeamSearchScorer' from 'transformers'` が出る場合は、Coqui TTSと互換性のないTransformers 5系が入っています。次で固定してください。

```powershell
pip install "transformers==4.33.3"
```

## 使い方

1. 参照音声を複数アップロードします。
2. 必要に応じて、各参照音声の重みを0〜100で調整します。
3. 言語を選び、話させたいセリフを入力します。
4. 音声利用の権限と本人同意を確認してチェックします。
5. `生成` を押して、プレビュー後にWAVをダウンロードします。

参照音声は、ノイズやBGMが少ない6〜30秒程度の自然な発話を推奨します。MP3などで失敗する場合はWAVに変換してください。

重みは、どの参照音声へ寄せるかを調整するための値です。内部的には重みに応じてXTTSへ渡す参照音声リストを展開するため、完全な連続値の話者埋め込み補間ではなく、モデル入力上の近似的な寄せ方になります。重みを変更した場合、リアルタイムVC用の `VC準備` は再実行してください。

## 音声置換

- `置換元のユーザー音声` に話し声ファイルを選び、`文字起こし` を押すとセリフ欄へ反映されます。
- `マイク` はブラウザの音声認識を使います。`マイクの確定発話を逐次生成` をオンにすると、確定した発話ごとに合成音声を作ります。

この方式はリアルタイム声質変換ではなく、STTからTTSへの置き換えです。元音声の息づかい、細かい抑揚、話速、重なった発話は完全には保持されません。

## リアルタイムVC

- `VC準備` は、参照音声群から一度だけ混合声のターゲット参照WAVを生成します。
- `VC変換` は、アップロードした置換元音声をFreeVC24で音声対音声変換します。
- `リアルタイムVC開始` は、マイク音声を約3秒ごとに区切って変換し、返ってきた音声を順番に再生します。

これはSTTを挟まないVCなので、元音声の言語内容、話すタイミング、ある程度の抑揚を保持します。ただしブラウザ録音チャンクをHTTPで逐次変換する構成のため、CPUでは実時間より遅くなることがあります。低遅延運用にはCUDA GPUを推奨します。

## 環境変数

- `VOICE_BLEND_FORCE_CPU=1`: GPUがあってもCPUで実行します。
- `VOICE_BLEND_MODEL`: 使用するCoqui TTSモデル名を変更します。
- `VOICE_BLEND_VC_MODEL`: 使用するCoqui VCモデル名です。既定値は `voice_conversion_models/multilingual/vctk/freevc24` です。
- `VOICE_BLEND_MAX_WEIGHT_REPETITIONS`: 重みに応じて参照音声を展開する最大反復数です。既定値は8です。
- `VOICE_BLEND_STT_MODEL`: Whisperの文字起こしモデル名です。既定値は `base` です。
- `VOICE_BLEND_PROFILE_DIR`: VCプロファイルの一時保存先です。
- `VOICE_BLEND_MAX_FILES`: 参照音声の最大数です。既定値は12です。
- `VOICE_BLEND_MAX_TOTAL_MB`: アップロード合計サイズ上限です。既定値は250MBです。

## 注意

このツールは、利用権限があり、本人同意のある音声だけに使ってください。生成物の公開や商用利用は、参照音声、モデル、生成内容の権利条件を確認してから行ってください。
