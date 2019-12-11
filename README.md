# Privacy-Sensing

Split audio:
https://unix.stackexchange.com/questions/280767/how-do-i-split-an-audio-file-into-multiple

Audio IO and using the proposed frame processing methods on audio data: _wav_read.py_

Load user field data and saved the processed data as class-based: _data_preparation_user.py_ (_wav_read.py, check_dir.py_)

Load class-based data (option 0 from _data_preparation_user.py_) and saved the processed data as fold-based: _concatenate_class_data.py_ (_wav_read.py, check_dir.py_)

Load (processed) user data and do mfcc extraction, 5-fold cross validation: _train_test_user.py_ (_wav_read.py_)

Enhance web collected audio (format conversion and volumn change): _audio_enhancement.py_

Calculate Mechanical Turk results: _read_mturk_files.py_

Test an audio clip with privacy protection: _wav_test.py_ (_wav_read.py_, _pca.py_)


