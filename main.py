import io
import logging
import time
from pathlib import Path

import librosa
import numpy as np
import soundfile

from sovits import infer_tool
from sovits import slicer
from sovits.infer_tool import Svc

logging.getLogger('numba').setLevel(logging.WARNING)
chunks_dict = infer_tool.read_temp("./sovits/chunks_temp.json")

model_name = "354_epochs.pth"  # 模型名称（pth文件夹下）
config_name = "config.json"
svc_model = Svc(f"./pth/{model_name}", f"./configs/{config_name}")
infer_tool.mkdir(["./raw", "./pth", "./results"])

# 支持多个wav文件，放在raw文件夹下
clean_names = ["时间煮雨"]
trans = [-3]  # 音高调整，支持正负（半音）
id_list = [1]  # 每次同时合成多序号音色
slice_db = -40  # 默认-40，嘈杂的音频可以-30，干声保留呼吸可以-50
wav_format = 'flac'  # 音频输出格式

infer_tool.fill_a_to_b(trans, clean_names)
for clean_name, tran in zip(clean_names, trans):
    raw_audio_path = f"raw/{clean_name}"
    if "." not in raw_audio_path:
        raw_audio_path += ".wav"
    infer_tool.format_wav(raw_audio_path)
    wav_path = Path(raw_audio_path).with_suffix('.wav')
    audio, sr = librosa.load(wav_path, mono=True, sr=None)
    wav_hash = infer_tool.get_md5(audio)
    if wav_hash in chunks_dict.keys():
        print("load chunks from temp")
        chunks = chunks_dict[wav_hash]["chunks"]
    else:
        chunks = slicer.cut(wav_path, db_thresh=slice_db)
    print(chunks)
    chunks_dict[wav_hash] = {"chunks": chunks, "time": int(time.time())}
    infer_tool.write_temp("./sovits/chunks_temp.json", chunks_dict)
    audio_data, audio_sr = slicer.chunks2audio(wav_path, chunks)

    for spk_id in id_list:
        audio = []
        for (slice_tag, data) in audio_data:
            print(f'#=====segment start, {round(len(data) / audio_sr, 3)}s======')
            length = int(np.ceil(len(data) / audio_sr * svc_model.target_sample))
            raw_path = io.BytesIO()
            soundfile.write(raw_path, data, audio_sr, format="wav")
            raw_path.seek(0)
            if slice_tag:
                print('jump empty segment')
                _audio = np.zeros(length)
            else:
                out_audio, out_sr = svc_model.infer(spk_id, tran, raw_path)
                _audio = out_audio.cpu().numpy()
            audio.extend(list(_audio))
        res_path = f'./results/{clean_name}_{tran}key_{svc_model.speakers[spk_id]}.{wav_format}'
        soundfile.write(res_path, audio, svc_model.target_sample, format=wav_format)
