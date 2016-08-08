#!/usr/bin/env python
import os
import platform
import re
import argparse

parser = argparse.ArgumentParser(prog=__file__)
parser.add_argument('-u', '--update', help="Update project, don't configure", action='store_true')

args, unknown = parser.parse_known_args()

if unknown:
    print(__file__ + " warning, ignore unknown options: " )
    for option in unknown:
        print(option)

def run(command):
    print(command)
    os.system(command)

def cmake_generator():
    host_os = platform.system()
    print("Detected " + host_os + " os")
    if host_os == "Windows" or re.match("CYGWIN_NT", host_os):
        GENERATOR="Visual Studio 14 Win64"
    elif host_os == "Linux":
        GENERATOR="Unix Makefiles"
    else:
        GENERATOR="Unknown"
        raise  OSError("Unsupported OS")
    return GENERATOR

def cmake(dir, copts = ""):
    os.chdir("../")
    if not os.path.exists(dir):
        os.makedirs(dir)
    os.chdir(dir)

    run('cmake' + copts + ' -G "' + cmake_generator() + '" ..')

    if not args.update:
        run('cmake-gui')

    os.chdir("..")

cmake("Build", "")
