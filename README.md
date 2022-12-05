# SoftVC VITS Singing Voice Conversion

## 模型简介（**main分支为sovits2.2（替换nsf声码器），2.0模型合成请下载2.0分支**）

歌声音色转换模型，通过Soft-VC内容编码器提取源音频语音特征，并提取音频f0，将两者结合输入VITS替换原本的文本输入达到歌声转换的效果。

## 使用方式（**实时sovits已更新完毕，适配多采样率的麦克风，音色热调节等**）

1、main.py为一键合成长时间音频（数分钟以上），环境安装参考vst版

2、gui开启127.0.0.1:7860网页，可在页面加载模型、转换语音（太长可能爆显存）

3、vst插件实时转换（慢半句左右）

3.0 git clone本项目，自行csdn安装cuda、torch的教程，注意与自己显卡的适配情况（cpu也行，就是很慢、效果差）

torch官网安装torch和相应的cuda：

https://pytorch.org/get-started/locally/

3.1 自行下载vst插件并搜索常用宿主软件的安装教程，宿主软件尽量装在c盘，反正不必要的bug

https://github.com/zhaohui8969/VST_NetProcess-/releases/tag/v1.2

目前经验，一定装在  C:\Program Files\Common Files\VST3\NetProcess.vst3（这是解压出的文件夹名字）

这样au、studio设置相应路径就能识别了，D盘不知道为什么不识别

3.2 c盘根目录新建/temp/vst文件夹，下载以下json放置在此文件夹（本项目configs里也有，直接复制省事）

https://github.com/zhaohui8969/VST_NetProcess-/blob/master/doc/netProcessConfig.json

"apiUrl": "http://127.0.0.1:6842"为默认api接口，与本程序flask_api.py默认端口对应

"speakId": "0","name": "猫雷"   分别为人物在模型中的id、人物名，**人物名使用英文**、否则乱码，插件中可以切换

3.3 项目根目录，新建pth文件夹，放入以下模型，模型配套本git的configs/nyarumul.json，flask_api默认参数就是这俩

此模型中序号—— 0：猫雷、2：云灏、3：即霜、4：奕兰秋

**这个是2.0模型，配合2.0分支的程序使用**

模型下载地址：https://huggingface.co/spaces/xiaolang/sovits_f0/resolve/main/152_epochs.pth

自行下载hubert-soft-0d54a1f4.pt改名为hubert.pt放置于pth文件夹下，一定要改名

hubert地址：https://github.com/bshall/hubert/releases/tag/v0.1

3.4 安装requirements.txt

3.5 首先运行！！！flask_api.py，待出现运行网址127.0.0.1:6842后，再打开vst插件（**项目的采样率一定要和麦克风的一样**）

插件有个小bug，必须等python的http成功运行后，才能调节插件的参数；不是大问题，自行注意即可

3.6 给某音轨挂载vst插件，打开录音准备、监听，录音输入增益可以调高一些，即可使用

3.7 插件的音调调到中间左右，默认是-30，正常使用是-15~+15；音频最小切片时间调到0，响应速度会更快

***************************************************************
soft-vc+vits拼接:@innnky

nsf-hifigan替换:@zhaohui8969
***************************************************************
Our codes are based on the following repos:

encoder:https://github.com/bshall/soft-vc

tts:https://github.com/jaywalnut310/vits

nsf-hifigan:https://github.com/openvpi/DiffSinger
