#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 03:25:44 2019

@author: dawei
"""

"""
Convert to 16-bit depth; modify volumn
"""

import ffmpeg
import subprocess

dir1 = './mturk/scenarios/phone/20191022_phone.m4a'
dir2 = './mturk/scenarios/phone/phone.wav'
# to change volumn:
# + " -filter:a 'volume=0' " \
command = "ffmpeg -i " + dir1 \
        + " -vn -sample_fmt s16 " \
        + " -ss 3 -to 21 " \
        + " -filter:a 'volume=1' " \
        + dir2
subprocess.call(command, shell=True)

