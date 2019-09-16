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

dir1 = './mturk/party.wav'
dir2 = './mturk/output.wav'
# to change volumn:
# + " -filter:a 'volume=0' " \
command = "ffmpeg -i " + dir1 \
        + " -vn -sample_fmt s16 " \
        + " -ss 0 -to 20 " \
        + dir2
subprocess.call(command, shell=True)

