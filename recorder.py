import pyaudio
import wave
input_filename = 1               # 麦克风采集的语音输入
input_filepath = "record/"              # 输入文件的path
import time
import threading
p = pyaudio.PyAudio()
FORMAT = pyaudio.paInt16
CHANNELS = 1                # 声道数
RATE = 22050                # 采样率
CHUNK = 256
stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)


def save_audio(WAVE_OUTPUT_FILENAME,CHANNELS,FORMAT,RATE,frames):
        wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
        
def get_audio(filepath,stream):
        #a=time.time() 
        CHUNK = 256
        RECORD_SECONDS = 2  # 录音时间
        WAVE_OUTPUT_FILENAME = filepath
        



        #print("*"*10, "开始录音：请在5秒内输入语音")
        frames = []
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)
        #print("*"*10, "录音结束\n")

        threading.Thread(target=save_audio, args=(WAVE_OUTPUT_FILENAME,CHANNELS,FORMAT,RATE,frames,)).start()
        #stream.stop_stream()
        #stream.close()
        #p.terminate()
        #print(time.time()-a)

        #print(time.time()-a)
# 联合上一篇博客代码使用，就注释掉下面，单独使用就不注释
while True:
    in_path = input_filepath + str(input_filename)+".wav"
    get_audio(in_path,stream)
    input_filename+=1
