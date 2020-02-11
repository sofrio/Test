# -*- coding: utf-8 -*-
__author__ = "kojima@sofrio.com"
__date__ = "Sep  8 15:05:31 2018"

import os
import sys
import shutil
import time
import datetime
import threading
import subprocess
import signal
import picamera
import RPi.GPIO as GPIO
import util as ut

CHECK_PERFORMANCE = False
SENSOR_TRIGGER = False if CHECK_PERFORMANCE else True
SENSE_DURATION = 5 #sec
RUN_MODE = "性能計測" if CHECK_PERFORMANCE else "センサー" if SENSOR_TRIGGER else "常時録画"

UPLOAD_META_DATA = False if CHECK_PERFORMANCE else True
UPLOAD_IMAGE = False if CHECK_PERFORMANCE else True
UPLOAD_SOUND = False if CHECK_PERFORMANCE else True
TRACE_UPLOAD = False if CHECK_PERFORMANCE else True

FRAME_SIZE = (320, 240)
INIT_FPS = 15 if SENSOR_TRIGGER else 10
FPS_ADJUST = 5
MIN_FPS = 5
MAX_FPS = 90 if CHECK_PERFORMANCE else 30

try:
    CAMERA_ID, PORT, LAT, LON = ut.file_get_contents("cameraID").split()
except IOError as err:
    print("cameraID ファイルにアクセスできません。(%s)" % err)
    sys.exit(-1)
except ValueError as err:
    print("cameraID ファイルが不正です。(%s)" % err)
    sys.exit(-1)

SENSOR_CHANNEL = 8
SENSE_DELTA = datetime.timedelta(seconds=SENSE_DURATION)
SENSE_INTERVAL = 0.5
SOUND_DURATION = 60 #sec
SHOW_RECORD_MSG = False
SHOW_ENCODE_MSG = False

BUFFER_DIR = "/tmp/cms/"
BUFFER_NAME = BUFFER_DIR + "buf"
BUFFER_SIZE = 60

SERVER_URL = "http://cms.japaneast.cloudapp.azure.com"
META_URL = SERVER_URL + ":60/cameramap/ipupdate.php"
IMAGE_URL = SERVER_URL + ":" + PORT + "/imageup.php"
SOUND_URL = SERVER_URL + ":" + PORT + "/soundup.php"

LOCAL_SERVER = False
if LOCAL_SERVER:
    CAMERA_ID = "28"
    PORT = "82"
    INIT_FPS = 30
    #FRAME_SIZE = (640, 480)　ダウンロードが間に合わない
    SERVER_URL = "http://192.168.1.60"
    META_URL = SERVER_URL + ":60/camerademo/ipupdate.php"
    SERVER_URL = "http://192.168.1.82"
    IMAGE_URL = SERVER_URL + ":" + PORT + "/imageup.php"
    SOUND_URL = SERVER_URL + ":" + PORT + "/soundup.php"
    SERVER_URL = "http://192.168.1.xx"


class Recorder(object):
    done = False # 全ての Recorder が同時に終了するためにクラス変数とする

    def __init__(self):
        self.event = threading.Event()
        self.wait = True
        self.wakes = 0
        self.wake_time = None
        self.record_times = datetime.timedelta(0)
        self.uploads = 0
        #print("Starting %s thread..." % type(self))
        self.thread = ut.threader(self.run)

    def run(self):
        while not Recorder.done:
            if self.check_wait(): continue
            try:
                if self.record():
                    self.stop()
            except Exception as ex:
                print(ex)

    def check_wait(self):
        if self.wait:
            if not self.event.wait(1): return True
            self.event.clear()
            self.wait = False
            self.wakes += 1
            self.wake_time = datetime.datetime.now()
        return False

    def record(self):
        class ProgramError(Exception):
            pass
        raise ProgramError("Recorder.record(self) must be overrided.")

    def start(self):
        if self.wait:
            self.event.set()

    def stop(self):
        if self.wait: return False
        self.wait = True
        self.record_times += datetime.datetime.now() - self.wake_time
        return True

    def upload_file(self, url, path, cap_time):
        if not os.path.exists(path): return
        self.uploads += 1
        time_str = cap_time.strftime("%Y-%m-%d-%H:%M:%S.%f")
        #print("upload_file(%s, %s)" % (path, time_str))
        if url:
            url += "?id=" + CAMERA_ID + "&time=" + time_str
            ut.upload_file(url, "file", path)
        else:
            if TRACE_UPLOAD:
                print("upload_file(%s, %s)" % (path, time_str))
        if os.path.exists(path):
            os.remove(path) # delete the jpg-file uploaded

    def terminate(self):
        Recorder.done = True
        self.stop()
        if self.thread:
            #print("waiting termination of %s thread." % type(self))
            self.thread.join()


class Audio(Recorder):
    def __init__(self):
        self.url = SOUND_URL if UPLOAD_SOUND else None
        self.rec_proc = None
        super(Audio, self).__init__()

    def record(self):
        rec = "arecord -D hw:1,0 -f S16_LE -d %d" % SOUND_DURATION
        self.rec_proc = subprocess.Popen(rec.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        start_time = datetime.datetime.now()
        path = BUFFER_DIR + start_time.strftime("%Y%m%d%H%M%S") + ".mp3"
        enc = "lame -b 128 - " + path
        enc_proc = subprocess.Popen(enc.split(), stdin=self.rec_proc.stdout, stderr=subprocess.PIPE)
        print(" >> %s 録音開始 (%s)" % (ut.pick_time(start_time), path))
        for rec_err in iter(self.rec_proc.stderr.readline, ''):
            if SHOW_RECORD_MSG or (rec_err.find("録音中") < 0 and rec_err.find("中断...") < 0):
                if rec_err.find("_snd_pcm_hw_open") >= 0 or rec_err.find("audio open error") >= 0:
                    print("エラー： マイクにアクセスできないので、録音はできません。")
                    return True
                print("arecord: " + rec_err)
        enc_proc.wait()
        ut.threader(self.upload_file, self.url, path, start_time)
        end_time = datetime.datetime.now()
        enc_err = enc_proc.stderr.read()
        if SHOW_ENCODE_MSG or enc_err.find("error") >= 0:
            print("lame: " + enc_err)
        self.rec_proc = None
        duration = (end_time - start_time).seconds
        print(" >> %s 録音終了 (%d 秒)" % (ut.pick_time(end_time), duration))
        return False

    def stop(self):
        super(Audio, self).stop()
        if self.rec_proc:
            ut.execute("pkill arecord")


class Buffer():
    buffers = []

    @staticmethod
    def clear():
        for name in os.listdir(BUFFER_DIR):
            path = BUFFER_DIR + name
            if os.path.isdir(path):
                shutil.rmtree(path)

    @staticmethod
    def get_avail():
        for buffer in Buffer.buffers:
            if buffer.is_avail(): return buffer
        buffer = Buffer(len(Buffer.buffers))
        Buffer.buffers.append(buffer)
        return buffer

    @staticmethod
    def file_count():
        files = 0
        for buffer in Buffer.buffers:
            files += len(os.listdir(buffer.dir))
        return files

    def __init__(self, nbr):
        self.nbr = nbr
        self.dir = "%s.%d/" % (BUFFER_NAME, self.nbr)
        if os.path.isdir(self.dir):
            for file in os.listdir(self.dir):
                os.remove(self.dir + file)
        else:
            os.makedirs(self.dir)
        self.url = IMAGE_URL if UPLOAD_IMAGE else None
        self.stop = False
        self.start_time = None
        self.end_time = None
        self.delta = 0

    def is_avail(self):
        return len(os.listdir(self.dir)) == 0

    def is_fixed(self):
        return self.end_time is not None

    def file_path(self, nbr):
        return self.dir + "cap%04d.jpg" % nbr

    def capture(self, cam):
        if not self.is_avail(): return 0
        self.stop = False
        self.end_time = None
        self.start_time = datetime.datetime.now()
        print(" == %s 録画開始 (%s)" % (ut.pick_time(self.start_time), self.dir))
        cam.capture_sequence(self.files(), use_video_port=True)
        self.end_time = datetime.datetime.now()
        files = len(os.listdir(self.dir))
        self.delta = (self.end_time - self.start_time) / files
        print(" == %s 録画終了 (%d 画像)" % (ut.pick_time(self.end_time), files))
        return files

    def files(self):
        for i in range(BUFFER_SIZE):
            if self.stop: break
            yield self.file_path(i)

    def upload(self, camera):
        if self.is_avail() or not self.is_fixed(): return
        cap_time = self.start_time
        for i in range(len(os.listdir(self.dir))):
            path = self.file_path(i)
            ut.threader(camera.upload_file, self.url, path, cap_time)
            cap_time += self.delta
        self.start_time = self.end_time = None


class Camera(Recorder):
    @staticmethod
    def create():
        try:
            return picamera.PiCamera()
        except Exception:
            print("\nカメラにアクセスできないので処理を終了します。")
            sys.exit(-1)

    def __init__(self, camera):
        self.camera = camera
        self.buffer = None
        self.remains = 0
        self.setup()
        super(Camera, self).__init__()

    def __del__(self):
        self.camera.close()

    def setup(self):
        self.camera.resolution = FRAME_SIZE
        self.camera.framerate = MAX_FPS - 5 if CHECK_PERFORMANCE else INIT_FPS
        self.camera.start_preview()
        time.sleep(2)
        #self.camera.hflip = False
        #self.camera.vflip = False
        self.camera.stop_preview()

    def record(self):
        self.adjust_fps()
        self.buffer = Buffer.get_avail()
        if self.buffer.capture(self.camera) < 0: return True
        start_time = self.buffer.start_time # self.bufferがすぐに変更されてしまうため
        ut.threader(self.meta_data, start_time)
        ut.threader(self.buffer.upload, self)
        self.buffer = None
        return False

    def adjust_fps(self):
        fps = self.camera.framerate
        msg = " ++ フレームレート：%d" % fps
        files = Buffer.file_count()
        delta = files - self.remains
        info = " [未送信画像： %d => %d (%+d)]" % (self.remains, files, delta)
        adjust = files - (BUFFER_SIZE + FPS_ADJUST)
        if files == 0 or (self.remains > 0 and abs(adjust) >= FPS_ADJUST):
            adjust = -1 if adjust > 0 else +1
            new_fps = min(MAX_FPS, max(MIN_FPS, fps + adjust))
            if files == 0:
                new_fps = max(INIT_FPS, new_fps) if CHECK_PERFORMANCE else INIT_FPS
            if new_fps != fps:
                def change_fps(camera, fps):
                    camera.framerate = fps
                ut.threader(change_fps, self.camera, new_fps)
                msg += " => %d" % new_fps
        print(msg + info)
        self.remains = files

    def meta_data(self, start_time):
        mtime = start_time.strftime("%Y-%m-%d-%H:%M:%S")
        '''
        query = "id=" + CAMERA_ID \
              + "&ip=" + "255.255.255.255" \
              + "&time=" + mtime \
              + "&lat=" + LAT \
              + "&lon=" + LON \
              + "&orient=" + str(0) \
              + "&angle=" + str(0) \
              + "&altitude=" + str(50) \
              + "&humidity=" + str(65) \
              + "&temperature=" + str(28.5) \
              + "&pressure=" + str(1024.0)
        '''
        query = "id=" + CAMERA_ID + "&time=" + mtime + "&lat=" + LAT + "&lon=" + LON
        url = "%s?%s" % (META_URL, query)
        if UPLOAD_META_DATA:
            ut.url_retrieve(url)
        else:
            if TRACE_UPLOAD:
                print("metadata: %s" % query)

    def stop(self):
        super(Camera, self).stop()
        if self.buffer:
            self.buffer.stop = True


class Sensor(object):
    def __init__(self, camera, audio):
        self.camera = camera
        self.audio = audio
        self.record_until = datetime.datetime.now()
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(SENSOR_CHANNEL, GPIO.IN)

    def sense(self):
        if GPIO.input(SENSOR_CHANNEL):
            sense_time = datetime.datetime.now()
            nl = "\n" if self.record_until < sense_time else ""
            self.record_until = sense_time + SENSE_DELTA
            if SENSOR_TRIGGER:
                self.camera.start()
                self.audio.start()
                start = ut.pick_time(sense_time)
                until = ut.pick_time(self.record_until)
                print(nl + "★赤外線感応： %s => %s" % (start, until))
        else:
            if SENSOR_TRIGGER:
                if self.record_until < datetime.datetime.now():
                    self.camera.stop()
                    self.audio.stop()
                    #time.sleep(0.1)
                    #ut.console_out(".")
        time.sleep(SENSE_INTERVAL)


def run(proc, *args):
    done = False
    def receive_signal(*_):
        print("\nシグナル検出...")
        run.done = True
    signal.signal(signal.SIGUSR1, receive_signal)
    print("Ctrl-C を押すか SIGUSR1 を送れば終了します。")
    try:
        while not done:
            proc(*args)
    except KeyboardInterrupt:
        print("\nCtrl-C　検出...")
        done = True

def main():
    cam = Camera.create()
    print("CMS camera#%s を%sモードで開始します。" % (CAMERA_ID, RUN_MODE))
    print("サーバーは %s です。" % SERVER_URL)
    Buffer.clear()
    audio = Audio()
    camera = Camera(cam)
    sensor = Sensor(camera, audio)
    if not SENSOR_TRIGGER:
        camera.start()
        audio.start()

    run(sensor.sense)

    camera.terminate()
    audio.terminate()
    #Buffer.clear()
    info = "計 %d 回で %s 録画しました。" % (camera.wakes, ut.omit_us(camera.record_times))
    print("\nCMS camera#%s を終了します... %s" % (CAMERA_ID, info))

def preview():
    print("Staring preview...")
    camera = picamera.PiCamera()
    camera.resolution = FRAME_SIZE
    camera.start_preview()

    run(time.sleep,60)

    camera.close()
    print("Preview finished.")

if __name__ == '__main__':
    ut.put_pid()
    PREVIEW = len(sys.argv) > 1 and sys.argv[1] == "-p"
    if PREVIEW: preview()
    else: main()
    os.remove(ut.PID_FILE)
