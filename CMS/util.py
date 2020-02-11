# -*- coding: utf-8 -*-
__author__ = "kojima@sofrio.com"
__date__ = "Sep  8 15:05:31 2018"

import os
import sys
import math
import termios
import socket
import thread
import threading
import subprocess

TMP_DIR = "/tmp/cms/"
PID_FILE = TMP_DIR + "pID"
SHOW_CURL_MSG = False
UTF8_BOM = bytearray([0xEF, 0XBB, 0XBF])


def file_get_contents(path):
    with open(path, 'r') as file:
        return file.read()

def file_put_contents(path, data):
    with open(path, 'w') as file:
        file.write(str(data))
    return data

def get_pid():
    return int(file_get_contents(PID_FILE))

def put_pid():
    if not os.path.isdir(TMP_DIR):
        os.mkdir(TMP_DIR)
    if os.path.exists(PID_FILE):
        #print("%s は既に存在します。" % PID_FILE)
        return None
    pid = os.getpid()
    return file_put_contents(PID_FILE, pid)

def sign(v):
    return +1 if v >=0 else -1

def largest_factor(product, max_factor):
    product = int(math.ceil(product))
    factor = 1
    for i in range(1, product):
        factor = int(math.ceil(product / float(i)))
        print(factor)
        if factor <= max_factor: break
    return factor

def get_cos(a, b):
    abcos = a[0] * b[0] + a[1] * b[1]
    a = math.sqrt(a[0] * a[0] + a[1] * a[1]) 
    b = math.sqrt(b[0] * b[0] + b[1] * b[1])
    ab = a * b
    return abcos / ab if ab else 1

def omit_us(datetime):
    datetimes = str(datetime).split('.')
    return datetimes[0]# + "." + datetimes[1][0]

def pick_time(datetime):
    return omit_us(datetime).split()[1]

def isatty(fd):
    try:
        termios.tcgetattr(fd)
    except termios.error:
        return False
    return True    # return True if no error

def console_out(msg):
    sys.stdout.write(msg)
    sys.stdout.flush()

def get_ip_adrs():
    hostname = socket.gethostname()
    return socket.gethostbyname_ex(hostname)

def threader(func, *args):
    try:
        task = threading.Thread(target=func, args=args)
        #task.setDaemon(True)
        task.start()
        return task
    except thread.error as err:
        print("%s: %s" % (err, str(func)))
        func(*args)
        return None

def shell(cmd):
    return subprocess.call(cmd, shell=True)

def execute(cmd):
    return subprocess.call(cmd.split())

def strip_bom(text):
    return text[len(UTF8_BOM):] if text.startswith(UTF8_BOM) else text

def curl(url):
    cmd = "curl " + url
    proc = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    proc.wait()
    out = strip_bom(proc.stdout.read())
    if out and SHOW_CURL_MSG:
        print("curl: %s" % out)
    err = strip_bom(proc.stderr.read()).split("curl")
    if len(err) > 1:
        print("curl(error): %s (%s)" % (err, url))
        return None
    return out

def url_retrieve(url):
    res = curl(url)
    if res is None: return True
    if res != "ok":
        print("curl: %s (%s)" % (res, url))
        return True
    return False

def upload_file(url, name, path):
    url += " -F " + name + "=@" + path
    res = curl(url)
    return res is None

if __name__ == '__main__':
    print(get_ip_adrs())
