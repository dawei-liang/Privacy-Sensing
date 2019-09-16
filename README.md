# Privacy-Sensing

Split audio:
https://unix.stackexchange.com/questions/280767/how-do-i-split-an-audio-file-into-multiple

Audio IO and using the proposed frame processing methods on audio data: _wav_read.py_

Load user field data and saved the processed data: _data_preparation_user.py_ (_wav_read.py, check_dir.py_)

Load (processed) user data and do mfcc extraction, 5-fold cross validation: _train_test_user.py_ (_wav_read.py_)

Enhance web collected audio (format conversion and volumn change): _audio_enhancement.py_
