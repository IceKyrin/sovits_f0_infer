import io
import logging

import maad
import numpy as np
import soundfile
import torchaudio
from flask import Flask, request, send_file
from flask_cors import CORS

from sovits.infer_tool import Svc

app = Flask(__name__)

CORS(app)

logging.getLogger('numba').setLevel(logging.WARNING)


class RealTimeVC:
    def __init__(self):
        self.last_chunk = None
        self.last_o = None
        self.chunk_len = 16000  # 区块长度
        self.pre_len = 3840  # 交叉淡化长度，640的倍数

    """输入输出都是1维numpy 音频波形数组"""

    def process(self, speaker_id, f_pitch_change, input_wav_path):
        audio, sr = torchaudio.load(input_wav_path)
        temp_wav = io.BytesIO()
        if self.last_chunk is None:
            soundfile.write(temp_wav, audio, sr, format="wav")
            audio, sr = svc_model.infer(speaker_id, f_pitch_change, temp_wav)
            audio = audio.cpu().numpy()
            self.last_chunk = wav[-self.pre_len:]
            self.last_o = audio
            return audio[-self.chunk_len:]
        else:
            audio = np.concatenate([self.last_chunk, wav])
            soundfile.write(temp_wav, audio, sr, format="wav")
            audio, sr = svc_model.infer(speaker_id, f_pitch_change, temp_wav)
            audio = audio.cpu().numpy()
            ret = maad.util.crossfade(self.last_o, audio, self.pre_len)
            self.last_chunk = wav[-self.pre_len:]
            self.last_o = audio
            return ret[self.chunk_len:2 * self.chunk_len]


@app.route("/voiceChangeModel", methods=["POST"])
def voice_change_model():
    request_form = request.form
    wave_file = request.files.get("sample", None)
    # 变调信息
    f_pitch_change = float(request_form.get("fPitchChange", 0))
    # DAW所需的采样率
    daw_sample = int(float(request_form.get("sampleRate", 0)))
    speaker_id = int(float(request_form.get("sSpeakId", 0)))
    # http获得wav文件并转换
    input_wav_path = io.BytesIO(wave_file.read())

    # 模型推理
    if raw_infer:
        out_audio, out_sr = svc_model.infer(speaker_id, f_pitch_change, input_wav_path)
        tar_audio = torchaudio.functional.resample(out_audio, svc_model.target_sample, daw_sample)
    else:
        out_audio = svc.process(speaker_id, f_pitch_change, input_wav_path)
        tar_audio = torchaudio.functional.resample(torch.form_numpy(out_audio), svc_model.target_sample, daw_sample)
    # 返回音频
    out_wav_path = io.BytesIO()
    soundfile.write(out_wav_path, tar_audio.cpu().numpy(), daw_sample, format="wav")
    out_wav_path.seek(0)
    return send_file(out_wav_path, download_name="temp.wav", as_attachment=True)


if __name__ == '__main__':
    # 启用则为直接切片合成，False为交叉淡化方式
    raw_infer = True
    # 每个模型和config是唯一对应的
    model_name = "524_epochs.pth"
    config_name = "config.json"
    svc_model = Svc(f"pth/{model_name}", f"configs/{config_name}")
    svc = RealTimeVC()
    # 此处与vst插件对应，不建议更改
    app.run(port=6842, host="0.0.0.0", debug=False, threaded=False)
