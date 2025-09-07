import os
import numpy as np
import librosa

def compute_jitter(audio, sr=22050):
    """计算音频的抖动（Jitter）"""
    f0 = librosa.yin(audio,  fmin=librosa.note_to_hz('C2'),  fmax=librosa.note_to_hz('C7')) 
    f0 = f0[f0 != 0]
    jitter = np.mean(np.abs(np.diff(f0)))  if len(f0) > 1 else 0
    return jitter

def load_audio_data(data_dir):
    """加载音频数据并提取特征"""
    audio_data = []
    labels = []
    
    for filename in os.listdir(data_dir): 
        if filename.endswith('.wav'): 
            file_path = os.path.join(data_dir,  filename)
            audio, sr = librosa.load(file_path,  sr=22050)
            
            # 提取MFCC特征及其一阶和二阶差分
            mfcc = librosa.feature.mfcc(y=audio,  sr=sr, n_mfcc=40)
            mfcc_delta = librosa.feature.delta(mfcc) 
            mfcc_delta2 = librosa.feature.delta(mfcc,  order=2)
            
            # 提取其他特征
            pitch = librosa.feature.rms(y=audio) 
            jitter = compute_jitter(audio, sr)
            cqt = librosa.feature.chroma_cqt(y=audio,  sr=sr)
            
            # 统一特征的高度和长度
            target_height, target_length = mfcc.shape 
            
            # 填充或截断特征以匹配MFCC的大小
            def pad_or_truncate(feature):
                pad_height = (target_height - feature.shape[0])  // 2
                pad_width = (target_length - feature.shape[1])  // 2
                feature_padded = np.pad( 
                    feature,
                    ((pad_height, pad_height), (pad_width, pad_width)),
                    mode='constant'
                )
                return feature_padded[:, :target_length]
            
            pitch = pad_or_truncate(pitch)
            jitter = np.full((target_height,  target_length), jitter)
            cqt = pad_or_truncate(cqt)
            
            # 将所有特征堆叠在一起
            features = np.stack([mfcc,  mfcc_delta, mfcc_delta2, pitch, jitter, cqt], axis=-1)
            features = np.expand_dims(features,  axis=0)
            audio_data.append(features) 
            
            # 提取标签
            label = int(filename.split('_')[0]) 
            labels.append(label) 
    
    return np.concatenate(audio_data,  axis=0), np.array(labels) 