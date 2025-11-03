import torch
import numpy as np
from scipy.io.wavfile import write

from audio.audio_processing import griffin_lim


def get_mel_from_wav(audio, _stft):
    audio = torch.clip(torch.FloatTensor(audio).unsqueeze(0), -1, 1)
    audio = torch.autograd.Variable(audio, requires_grad=False)
    melspec, energy = _stft.mel_spectrogram(audio)
    melspec = torch.squeeze(melspec, 0).numpy().astype(np.float32)
    energy = torch.squeeze(energy, 0).numpy().astype(np.float32)

    return melspec, energy


def inv_mel_spec(mel, out_filename, _stft, griffin_iters=60):
    mel = torch.stack([mel])
    mel_decompress = _stft.spectral_de_normalize(mel)
    mel_decompress = mel_decompress.transpose(1, 2).data.cpu()
    spec_from_mel_scaling = 1000
    spec_from_mel = torch.mm(mel_decompress[0], _stft.mel_basis)
    spec_from_mel = spec_from_mel.transpose(0, 1).unsqueeze(0)
    spec_from_mel = spec_from_mel * spec_from_mel_scaling

    audio = griffin_lim(
        torch.autograd.Variable(spec_from_mel[:, :, :-1]), _stft._stft_fn, griffin_iters
    )

    audio = audio.squeeze()
    audio = audio.cpu().numpy()
    audio_path = out_filename
    write(audio_path, _stft.sampling_rate, audio)


import numpy as np

def spec2ceps(spec, order=None):
    """
    Args:
        spec (np.ndarray): スペクトル(N,)またはスペクトログラム(N, M)。
                           Nは周波数ビンの数 (MATLABの size(spec, 1))。
                           入力が1D配列(N,)の場合、(N, 1)の列ベクトルとして扱われます。
                           
        order (int, optional): リフタリング次数（保持するケプストラムの総数）。
                               MATLABの `order` に相当します。
                               Noneの場合、非冗長な全成分 (N) を返します。

    Returns:
        np.ndarray: ケプストラムまたはケプストログラム。
                    入力が1Dだった場合は1Dで、2Dだった場合は2Dで返されます。
    """
    
    # 1. 入力の次元をチェック
    # NumPyでは1D配列(N,)が一般的ですが、MATLABのコードは(N, 1)や(N, M)を
    # 前提としているため、処理を統一するために2Dに揃えます。
    was_1d = False
    if spec.ndim == 1:
        was_1d = True
        # (N,) -> (N, 1) に変形
        spec = spec.reshape(-1, 1)

    # 2. スペクトルのポイント数（周波数ビン数？）を取得
    len_spec = spec.shape[0]
    # 3. 対数スペクトル
    specgLog = np.log(spec)

    # 4. 対数スペクトルを左右対称に復元
    # specgLog = [specgLog; flip(specgLog(2:len - 1, :))]; (MATLAB)
    
    # MATLABの 2:len-1 (1-based index) は、
    # Pythonの 1:len_spec-1 (0-based index) に相当します。
    # (直流0Hzとナイキスト周波数成分を除いた部分)
    to_flip = specgLog[1:len_spec - 1, :]
    
    # flip(...) (MATLAB) -> np.flip(..., axis=0) (Python)
    # axis=0 で上下（周波数軸）に反転します。
    flipped_part = np.flip(to_flip, axis=0)
    
    # [A; B] (MATLAB) -> np.concatenate((A, B), axis=0) (Python)
    # 縦に連結
    specgLog_full = np.concatenate((specgLog, flipped_part), axis=0)

    # 5. ケプストラムの計算 (IFFT)
    # n = 2*(len - 1)
    n_fft = 2 * (len_spec - 1)
    
    # cepsComplex = ifft(specgLog, n_fft, 1); (MATLAB)
    # MATLABの dim=1 (1次元目=行) は、Pythonの axis=0 に相当します。
    cepsComplex = np.fft.ifft(specgLog_full, n=n_fft, axis=0)

    # 6. 実数部の取得
    # ceps = real(cepsComplex); (MATLAB)
    ceps = np.real(cepsComplex)

    # 7. リフタリング
    # Pythonでは配列の削除(MATLABの `A(x:end, :) = []`) よりも、
    # 保持したい部分をスライス(`A = A[:x, :]`)で指定するのが一般的です。
    
    if order is not None:
        # if nargin == 2 (MATLAB)
        # ceps(order + 1:end, :) = []; (MATLAB)
        # -> 1からorderまでの要素(order個)を保持します。
        # Python: 0からorder-1までの要素(order個)を保持します。
        ceps_liftered = ceps[:order, :]
    else:
        # else (MATLAB)
        # ceps(len + 1:end, :) = []; (MATLAB)
        # -> 1からlen_specまでの要素(len_spec個)を保持します。
        # (IFFTの結果の非冗長な部分)
        # Python: 0からlen_spec-1までの要素(len_spec個)を保持します。
        ceps_liftered = ceps[:len_spec, :]

    # 8. 元の次元に戻す
    if was_1d:
        # (N, 1) -> (N,) に戻す
        return ceps_liftered.squeeze(axis=1)
    else:
        # 2D (N, M) のまま返す
        return ceps_liftered

def ceps2dCeps(ceps, K):
    """
    ケプストラム（ケプストログラム）からΔケプストラム（の時系列）を計算
    
    Args:
        ceps (np.ndarray): ケプストラム（の時系列）。
                           サイズは (sampleL, timeL)。
                           sampleL: ケフレンシー軸のポイント数（リフタリング次数に相当）
                           timeL: 時間軸のポイント数
        K (int): 線形単回帰に用いる時間幅を決定するパラメータ [ポイント]
    
    Returns:
        np.ndarray: Δケプストラム（の時系列）。
                    サイズは (sampleL, timeL - 2*K)。
    """
    
    # ケプストラムのポイント数
    sampleL = ceps.shape[0]  # ケフレンシー軸のポイント数（リフタリング次数に相当）
    timeL = ceps.shape[1]    # 時間軸のポイント数
    
    # Δケプストラムの計算（詳細は数式参照）
    dCeps = np.zeros((sampleL, timeL - 2*K))
    k = np.arange(-K, K + 1)  # 時間幅
    
    i = 0  # フレーム番号
    for t in range(K, timeL - K):  # フレームシフト
        # 分子: sum(k.*ceps(:, t + k), 2)
        # k と ceps[:, t + k] の要素ごとの積を、時間軸（axis=1）で合計
        numerator = np.sum(k * ceps[:, t + k], axis=1)
        
        # 分母: sum(k.^2, 2)
        denominator = np.sum(k**2)
        
        # Δケプストラム
        dCeps[:, i] = numerator / denominator
        
        i += 1
    
    return dCeps

def dCeps2norm(dCeps, isPower=True):
    """
    Δケプストラム（の時系列）からΔケプストラムのノルム（の時系列）を計算
    
    Args:
        dCeps (np.ndarray): Δケプストラム（の時系列）。
                            サイズは (sampleL, timeL)。
                            sampleL: ケフレンシー軸のポイント数
                            timeL: 時間軸のポイント数
        isPower (bool): True: パワー成分を含む, False: パワー成分を含まない
                        デフォルトはTrue
    
    Returns:
        np.ndarray: ノルム（ベクトル空間における長さ）の時系列。
                    サイズは (timeL,)。
    """
    
    # 20/log(10)で底をeから10に変換してデシベル値を求める
    # パワー成分を含む
    if isPower:
        # dCeps[0, :] は第1成分（パワー成分）
        # dCeps[1:, :] は第2成分以降
        # sum(..., axis=0) で各時刻ごとに合計
        norm = (20 / np.log(10)) * np.sqrt(
            2 * np.sum(dCeps[1:, :] ** 2, axis=0) + dCeps[0, :] ** 2
        )
    # パワー成分を含まない
    else:
        norm = (20 / np.log(10)) * np.sqrt(
            2 * np.sum(dCeps[1:, :] ** 2, axis=0)
        )
    
    return norm


def get_DF_from_wav(audio, _stft, order=32, K=2):
    """
    音声波形からDF時系列を計算する関数
    
    Args:
        audio (np.ndarray): 音声波形
        _stft: STFTクラスのインスタンス
        order (int): リフタリング次数（デフォルト: 32）
        K (int): 線形単回帰に用いる時間幅パラメータ（デフォルト: 2）
    
    Returns:
        np.ndarray: DF時系列
    """
    # 1. 前処理：audioをclipしてFloatTensorに変換
    audio = torch.clip(torch.FloatTensor(audio).unsqueeze(0), -1, 1)
    audio = torch.autograd.Variable(audio, requires_grad=False)
    
    # 2. _stftのtransformメソッドから振幅スペクトログラムを取得
    magnitude, phase = _stft.transform(audio)
    
    # 3. バッチ次元を削除してNumPy配列に変換
    magnitude = torch.squeeze(magnitude, 0).numpy().astype(np.float32)
    
    # 4. spec2cepsによってケプストログラムに変換
    cepstrogram = spec2ceps(magnitude, order=order)
    
    # 5. ceps2dCepsによってΔケプストログラムに変換
    delta_cepstrogram = ceps2dCeps(cepstrogram, K=K)
    
    # 6. dCeps2normによってDF時系列を得る
    df_series = dCeps2norm(delta_cepstrogram, isPower=True)
    
    # 7. DF時系列を返す
    return df_series