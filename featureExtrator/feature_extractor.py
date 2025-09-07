import librosa
import numpy as np

def extract_mfcc(audio, sr=22050, n_mfcc=40):
    """提取MFCC特征"""
    return librosa.feature.mfcc(y=audio,  sr=sr, n_mfcc=n_mfcc)

def extract_mfcc_delta(mfcc):
    """提取MFCC的一阶差分"""
    return librosa.feature.delta(mfcc) 

def extract_mfcc_delta2(mfcc):
    """提取MFCC的二阶差分"""
    return librosa.feature.delta(mfcc,  order=2)

def extract_chroma_cqt(audio, sr=22050):
    """提取Chroma CQT特征"""
    return librosa.feature.chroma_cqt(y=audio,  sr=sr)

def extract_rms(audio):
    """提取RMS能量特征"""
    return librosa.feature.rms(y=audio) 