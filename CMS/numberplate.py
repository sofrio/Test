# -*- coding: utf-8 -*-
__author__ = "kojima@sofrio.com"
__date__ = "Sep  21 11:12:42 2018"

#追加

import os
import sys
import time
import datetime
import signal
import random
import math
import RPi.GPIO as GPIO
import util as ut

CHECK_PERFORMANCE = False
RUN_MODE = "性能計測" if CHECK_PERFORMANCE else "通常追跡"

UPLOAD_META_DATA = False if CHECK_PERFORMANCE else True
TRACE_UPLOAD = False if CHECK_PERFORMANCE else True

try:
    GPS_ID, LAT, LON = ut.file_get_contents("numberID").split()
except IOError as err:
    print("numberID ファイルにアクセスできません。(%s)" % err)
    sys.exit(-1)
except ValueError as err:
    print("numberID ファイルが不正です。(%s)" % err)
    sys.exit(-1)

GPS_DEV = "/dev/ttyACM0"
SEND_INTERVAL = 1

SERVER_URL = "http://cms.japaneast.cloudapp.azure.com"
META_URL = SERVER_URL + ":60/numberplate/ipupdate.php"

# お遊び
BULK_SEND = True
TARGET_LAT = 35.633
TARGET_LON = 139.484
TARGET_RANGE = ((-0.12, 0.03), (-0.28, 0.06))
TARGET_BASE = 0.05
DURATION = 300 # sec
MIN_STEP = 0.001
RANDOM_RATIO = 1.5
MAX_STRAIGHT = 6
DELTA_RATIO = 0.05

class GPS(object):
    def __init__(self):
        self.pos = (float(LAT), float(LON))
        self.target = None
        self.target_delta = None
        self.target_dir = None
        self.range = None
        self.delta = None
        self.straights = MAX_STRAIGHT // 2
        self.setup_target()
        self.sends = 0
        self.now = datetime.datetime.now()

    def setup_target(self):
        if self.target is None:
            self.target = [TARGET_LAT, TARGET_LON]
            self.target_delta = [t - p for t, p in zip(self.target, self.pos)]
        else:
            cos = -1
            while cos <= 0:
                delta = [random.uniform(l, u) for l, u in TARGET_RANGE]
                #delta = [d + TARGET_BASE if d >=0 else -TARGET_BASE for d in delta]
                cos = ut.get_cos(self.target_delta, delta)
            self.target = [p + d for p, d in zip(self.pos, delta)]
            self.target_delta = delta
        self.target_dir = [d > 0 for d in self.target_delta]
        step = [d / DURATION * SEND_INTERVAL for d in self.target_delta]
        max_step = max(MIN_STEP, max(abs(step[0]), abs(step[1])))
        lower = [(s - max_step * RANDOM_RATIO) / RANDOM_RATIO for s in step]
        upper = [(s + max_step * RANDOM_RATIO) / RANDOM_RATIO for s in step]
        self.range = zip(lower, upper)
        if self.delta is None:
            self.delta = [s / 2 for s in step]
        print("New target = (%f, %f)" % (self.target[0], self.target[1]))

    def meta_data(self):
        if BULK_SEND:
            now = self.now
            self.now += datetime.timedelta(seconds=1)
            done = self.update_pos()
            self.send_data(now)
            if done:
                self.setup_target()
            return
        now = int(time.time())
        if self.sends == 0 or (now % SEND_INTERVAL) == 0:
            now = datetime.datetime.now()
            done = self.update_pos()
            ut.threader(self.send_data, now)
            self.sends += 1
            if done:
                self.setup_target()
            time.sleep(1)

    def update_pos(self):
        if self.straights > 0:
            delta = [random.uniform(l, u) * DELTA_RATIO for l, u in self.range]
            delta = [s + d for s, d in zip(self.delta, delta)]
            self.straights -= 1
        else:
            isqrt2 = 1. / math.sqrt(2) # cos(pi/4)
            cos = -1
            while cos < isqrt2:
                delta = [random.uniform(l, u) for l, u in self.range]
                cos = ut.get_cos(self.delta, delta)
            self.straights = random.randrange(0, MAX_STRAIGHT)
        self.delta = delta
        self.pos = [p + d for p, d in zip(self.pos, self.delta)]
        return all([(p > t) == d for p, t, d in zip(self.pos, self.target, self.target_dir)])

    def send_data(self, send_time):
        mtime = send_time.strftime("%Y-%m-%d-%H:%M:%S")
        query = "id=" + GPS_ID \
              + "&time=" + mtime \
              + "&lat=" + str(self.pos[0]) \
              + "&lon=" + str(self.pos[1]) \
              + "&orient=" + str(0) \
              + "&angle=" + str(0) \
              + "&altitude=" + str(50) \
              + "&humidity=" + str(65) \
              + "&temperature=" + str(28.5) \
              + "&pressure=" + str(1024.0)
        url = "%s?%s" % (META_URL, query)
        if UPLOAD_META_DATA:
            ut.url_retrieve(url)
            if TRACE_UPLOAD:
                print("%s (%f, %f)" % (mtime, self.pos[0], self.pos[1]))
        else:
            if TRACE_UPLOAD:
                print("metadata: %s" % query)


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
    print("NPTS GPS#%s を%sモードで開始します。" % (GPS_ID, RUN_MODE))
    print("サーバーは %s です。" % SERVER_URL)
    gps = GPS()

    run(gps.meta_data)

    info = "計 %d 回送信しました。" % gps.sends
    print("\nNPTS GPS#%s を終了します... %s" % (GPS_ID, info))


if __name__ == '__main__':
    ut.put_pid()
    main()
    os.remove(ut.PID_FILE)
