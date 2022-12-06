# coding=utf-8
import logging

import gradio as gr
import soundfile
import torch

from sovits import infer_tool
from sovits.infer_tool import Svc

logging.getLogger('numba').setLevel(logging.WARNING)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

spk_dict = {}

svc_model = None


def load_model(md_name, cfg_json):
    global svc_model, spk_dict
    svc_model = Svc(md_name, cfg_json)
    if "speakers" in svc_model.hps_ms.keys():
        for i, spk in enumerate(svc_model.hps_ms.speakers):
            spk_dict[spk] = i
        spk_list = list(spk_dict.keys())
    else:
        spk_list = ["0"]
    return "模型加载成功", gr.Dropdown.update(choices=spk_list)


def infer(sid, audio_record, audio_upload, tran):
    if audio_upload is not None:
        audio_path = audio_upload
    elif audio_record is not None:
        audio_path = audio_record
    else:
        return "你需要上传wav文件或使用网页内置的录音！", None
    o_audio, out_sr = svc_model.infer(spk_dict[sid], tran, audio_path)
    o_audio = o_audio.cpu().numpy()
    out_path = f"./out_temp.wav"
    soundfile.write(out_path, o_audio, svc_model.target_sample)
    return svc_model.target_sample, o_audio


app = gr.Blocks()
with app:
    with gr.Tabs():
        with gr.TabItem("合成"):
            gr.Markdown(value="""
            本模型为sovits_f0，支持**45s以内**的**无伴奏**wav、mp3、aac、m4a格式，或使用**网页内置**的录音（二选一）    
            """)
            model_name = gr.Dropdown(label="模型", choices=infer_tool.get_end_file("./pth", "pth"))
            config_json = gr.Dropdown(label="配置", choices=infer_tool.get_end_file("./configs", "json"))
            vc_config = gr.Button("加载模型", variant="primary")
            model_mess = gr.Textbox(label="Output Message")

            with gr.Box():
                speaker_id = gr.Dropdown(label="目标音色")
                record_input = gr.Audio(source="microphone", label="录制你的声音", type="filepath",
                                        elem_id="audio_inputs")
                upload_input = gr.Audio(source="upload", label="上传音频（长度小于45秒）", type="filepath",
                                        elem_id="audio_inputs")
                vc_transform = gr.Number(label="变调（整数，可以正负，半音数量，升高八度就是12）", value=0)
                vc_submit = gr.Button("转换", variant="primary")
                out_audio = gr.Audio(label="Output Audio")
            vc_config.click(load_model, [model_name, config_json], [model_mess, speaker_id])
            vc_submit.click(infer, [speaker_id, record_input, upload_input, vc_transform],
                            [out_audio])
        with gr.TabItem("使用说明"):
            gr.Markdown(value="""
                        0、合集：https://github.com/IceKyrin/sovits_guide/blob/main/README.md
                        
                        1、仅支持sovit_f0（sovits2.0）模型
                        
                        2、自行下载hubert-soft-0d54a1f4.pt改名为hubert.pt放置于pth文件夹下（已经下好了）
                            https://github.com/bshall/hubert/releases/tag/v0.1

                        3、pth文件夹下放置sovits2.0的模型
                        
                        4、与模型配套的xxx.json，需有speaker项——人物列表
                        
                        5、放无伴奏的音频、或网页内置录音，不要放奇奇怪怪的格式
                        
                        6、仅供交流使用，不对用户行为负责

                        """)
    app.launch()
