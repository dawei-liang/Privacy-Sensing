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


Partially adopted from: https://github.com/anuragkr90/weak_feature_extractor, by Kumar et al., ICASSP 18

Load the esc wav files, degrade, and save the degraded mfcc frames(csv) or full wav clips(wav): _load_esc.py (wav_read.py)_

Load processed wav, compute segment spectrogram, extract embedding features, and save as csv (require torch): _feat_extractor.py (extractor.py, network_architectures.py)_

Evaluation for esc features: 
for embedding features from transfer leanring: _train_test_esc_tl.py (check_dirs.py)_
for mfcc features:  _train_test_esc_mfcc.py (check_dirs.py)_
