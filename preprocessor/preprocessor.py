import os
import random
import json

import tgt
import librosa
import numpy as np
import pyworld as pw
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import audio as Audio


class Preprocessor:
    def __init__(self, config):
        self.config = config
        self.in_dir = config["path"]["raw_path"]
        self.out_dir = config["path"]["preprocessed_path"]
        self.val_size = config["preprocessing"]["val_size"]
        self.sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
        self.hop_length = config["preprocessing"]["stft"]["hop_length"]

        assert config["preprocessing"]["pitch"]["feature"] in [
            "phoneme_level",
            "frame_level",
        ]
        assert config["preprocessing"]["energy"]["feature"] in [
            "phoneme_level",
            "frame_level",
        ]
        self.pitch_phoneme_averaging = (
            config["preprocessing"]["pitch"]["feature"] == "phoneme_level"
        )
        self.energy_phoneme_averaging = (
            config["preprocessing"]["energy"]["feature"] == "phoneme_level"
        )

        self.pitch_normalization = config["preprocessing"]["pitch"]["normalization"]
        self.energy_normalization = config["preprocessing"]["energy"]["normalization"]
        
        # DFの有効/無効フラグ
        self.df_enable = config["preprocessing"]["df"]["enable"]
        
        if self.df_enable:
            assert config["preprocessing"]["df"]["feature"] in [
                "phoneme_level",
                "frame_level",
            ]
            self.df_phoneme_averaging = (
                config["preprocessing"]["df"]["feature"] == "phoneme_level"
            )
            self.df_normalization = config["preprocessing"]["df"]["normalization"]

        self.STFT = Audio.stft.TacotronSTFT(
            config["preprocessing"]["stft"]["filter_length"],
            config["preprocessing"]["stft"]["hop_length"],
            config["preprocessing"]["stft"]["win_length"],
            config["preprocessing"]["mel"]["n_mel_channels"],
            config["preprocessing"]["audio"]["sampling_rate"],
            config["preprocessing"]["mel"]["mel_fmin"],
            config["preprocessing"]["mel"]["mel_fmax"],
        )
        
        # DF計算用の別のSTFTインスタンス（DFが有効な場合のみ）
        if self.df_enable:
            self.STFT_DF = Audio.stft.STFT(
                config["preprocessing"]["stft"]["filter_length"],
                config["preprocessing"]["stft"]["hop_length"],
                config["preprocessing"]["stft"]["win_length"],
            )

    def build_from_path(self):
        os.makedirs((os.path.join(self.out_dir, "mel")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "pitch")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "energy")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "duration")), exist_ok=True)
        if self.df_enable:
            os.makedirs((os.path.join(self.out_dir, "df")), exist_ok=True)

        print("Processing Data ...")
        out = list()
        n_frames = 0
        pitch_scaler = StandardScaler()  # 標準化のためのスケーラー
        energy_scaler = StandardScaler()  # 標準化のためのスケーラー
        if self.df_enable:
            df_scaler = StandardScaler()  # 標準化のためのスケーラー

        # Compute pitch, energy, duration, and mel-spectrogram
        speakers = {}
        for i, speaker in enumerate(tqdm(os.listdir(self.in_dir))):
            speakers[speaker] = i  # LJSpeechの場合は、speakerはLJSpeech, iは0
            for wav_name in os.listdir(os.path.join(self.in_dir, speaker)):
                if ".wav" not in wav_name:  # wavファイルでない場合はスキップ
                    continue

                basename = wav_name.split(".")[0]
                tg_path = os.path.join(
                    self.out_dir, "TextGrid", speaker, "{}.TextGrid".format(basename)  # {}はbasenameに置き換えられる
                )
                if os.path.exists(tg_path):
                    ret = self.process_utterance(speaker, basename)
                    if ret is None:
                        continue
                    else:
                        if self.df_enable:
                            info, pitch, energy, df, n = ret # infoはメタデータ、pitchはピッチ、energyはエネルギー、dfはDF、nはフレーム数
                        else:
                            info, pitch, energy, n = ret # DFなしの場合
                    out.append(info)

                if len(pitch) > 0:
                    pitch_scaler.partial_fit(pitch.reshape((-1, 1)))
                if len(energy) > 0:
                    energy_scaler.partial_fit(energy.reshape((-1, 1)))
                if self.df_enable and len(df) > 0:
                    df_scaler.partial_fit(df.reshape((-1, 1)))

                n_frames += n

        print("Computing statistic quantities ...")
        # Perform normalization if necessary
        if self.pitch_normalization:
            pitch_mean = pitch_scaler.mean_[0]
            pitch_std = pitch_scaler.scale_[0]
        else:
            # A numerical trick to avoid normalization...
            pitch_mean = 0
            pitch_std = 1
        if self.energy_normalization:
            energy_mean = energy_scaler.mean_[0]
            energy_std = energy_scaler.scale_[0]
        else:
            energy_mean = 0
            energy_std = 1

        pitch_min, pitch_max = self.normalize(
            os.path.join(self.out_dir, "pitch"), pitch_mean, pitch_std
        )
        energy_min, energy_max = self.normalize(
            os.path.join(self.out_dir, "energy"), energy_mean, energy_std
        )
        
        if self.df_enable:
            if self.df_normalization:
                df_mean = df_scaler.mean_[0]
                df_std = df_scaler.scale_[0]
            else:
                df_mean = 0
                df_std = 1
            df_min, df_max = self.normalize(
                os.path.join(self.out_dir, "df"), df_mean, df_std
            )

        # Save files
        with open(os.path.join(self.out_dir, "speakers.json"), "w") as f:
            f.write(json.dumps(speakers))

        with open(os.path.join(self.out_dir, "stats.json"), "w") as f:
            stats = {
                "pitch": [
                    float(pitch_min),
                    float(pitch_max),
                    float(pitch_mean),
                    float(pitch_std),
                ],
                "energy": [
                    float(energy_min),
                    float(energy_max),
                    float(energy_mean),
                    float(energy_std),
                ],
            }
            if self.df_enable:
                stats["df"] = [
                    float(df_min),
                    float(df_max),
                    float(df_mean),
                    float(df_std),
                ]
            f.write(json.dumps(stats))

        print(
            "Total time: {} hours".format(
                n_frames * self.hop_length / self.sampling_rate / 3600
            )
        )

        random.shuffle(out)
        out = [r for r in out if r is not None]

        # Write metadata
        with open(os.path.join(self.out_dir, "train.txt"), "w", encoding="utf-8") as f:
            for m in out[self.val_size :]:
                f.write(m + "\n")
        with open(os.path.join(self.out_dir, "val.txt"), "w", encoding="utf-8") as f:
            for m in out[: self.val_size]:
                f.write(m + "\n")

        return out

    def process_utterance(self, speaker, basename):
        """
        1つの発話データを処理し、音響特徴量を抽出・保存する
        
        Args:
            speaker (str): 話者名
            basename (str): 音声ファイルのベース名（拡張子なし）
            
        Returns:
            tuple or None: 処理成功時は (メタデータ文字列, ピッチ配列, エネルギー配列, フレーム数) を返す。
                          処理失敗時は None を返す。
        """
        wav_path = os.path.join(self.in_dir, speaker, "{}.wav".format(basename))
        text_path = os.path.join(self.in_dir, speaker, "{}.lab".format(basename))
        tg_path = os.path.join(
            self.out_dir, "TextGrid", speaker, "{}.TextGrid".format(basename)
        )

        # Get alignments
        textgrid = tgt.io.read_textgrid(tg_path)
        phone, duration, start, end = self.get_alignment(
            textgrid.get_tier_by_name("phones")  # "phones"はTextGridのtier名(つまり音素の層を取得)
        )
        text = "{" + " ".join(phone) + "}"
        if start >= end:
            return None

        # Read and trim wav files
        wav, _ = librosa.load(wav_path)  # wav_pathの音声ファイルを読み込み、wavに格納、_はサンプリングレート(デフォルトは22050)
        wav = wav[
            int(self.sampling_rate * start) : int(self.sampling_rate * end)
        ].astype(np.float32)
        # 音声データを開始時刻から終了時刻まで切り取り、float32型に変換

        # Read raw text
        with open(text_path, "r") as f:
            raw_text = f.readline().strip("\n")

        # Compute fundamental frequency
        pitch, t = pw.dio(
            wav.astype(np.float64), 
            self.sampling_rate,
            frame_period=self.hop_length / self.sampling_rate * 1000,
        )
        pitch = pw.stonemask(wav.astype(np.float64), pitch, t, self.sampling_rate)

        pitch = pitch[: sum(duration)]
        if np.sum(pitch != 0) <= 1:
            return None

        # Compute mel-scale spectrogram and energy
        mel_spectrogram, energy = Audio.tools.get_mel_from_wav(wav, self.STFT)
        mel_spectrogram = mel_spectrogram[:, : sum(duration)]
        energy = energy[: sum(duration)]
        
        # Compute Dynamic Feature (DF) - 有効な場合のみ
        if self.df_enable:
            df = Audio.tools.get_DF_from_wav(wav, self.STFT_DF)
            # get_DF_from_wav内でパディングされ、元のスペクトログラムと同じフレーム数になる
            # durationに合わせて必要な部分を抽出
            df = df[: sum(duration)]

        if self.pitch_phoneme_averaging:
            # perform linear interpolation
            nonzero_ids = np.where(pitch != 0)[0]
            interp_fn = interp1d(
                nonzero_ids,
                pitch[nonzero_ids],
                fill_value=(pitch[nonzero_ids[0]], pitch[nonzero_ids[-1]]),
                bounds_error=False,
            )
            pitch = interp_fn(np.arange(0, len(pitch)))

            # Phoneme-level average
            pos = 0
            for i, d in enumerate(duration):
                if d > 0:
                    pitch[i] = np.mean(pitch[pos : pos + d])
                else:
                    pitch[i] = 0
                pos += d
            pitch = pitch[: len(duration)]

        if self.energy_phoneme_averaging:
            # Phoneme-level average
            pos = 0
            for i, d in enumerate(duration):
                if d > 0:
                    energy[i] = np.mean(energy[pos : pos + d])
                else:
                    energy[i] = 0
                pos += d
            energy = energy[: len(duration)]
        
        # DF phoneme-level averaging - 有効な場合のみ
        if self.df_enable and self.df_phoneme_averaging:
            # Phoneme-level average
            pos = 0
            for i, d in enumerate(duration):
                if d > 0:
                    df[i] = np.mean(df[pos : pos + d])
                else:
                    df[i] = 0
                pos += d
            df = df[: len(duration)]

        # Save files
        dur_filename = "{}-duration-{}.npy".format(speaker, basename)
        np.save(os.path.join(self.out_dir, "duration", dur_filename), duration)

        pitch_filename = "{}-pitch-{}.npy".format(speaker, basename)
        np.save(os.path.join(self.out_dir, "pitch", pitch_filename), pitch)

        energy_filename = "{}-energy-{}.npy".format(speaker, basename)
        np.save(os.path.join(self.out_dir, "energy", energy_filename), energy)

        if self.df_enable:
            df_filename = "{}-df-{}.npy".format(speaker, basename)
            np.save(os.path.join(self.out_dir, "df", df_filename), df)

        mel_filename = "{}-mel-{}.npy".format(speaker, basename)
        np.save(
            os.path.join(self.out_dir, "mel", mel_filename),
            mel_spectrogram.T,
        )

        if self.df_enable:
            return (
                "|".join([basename, speaker, text, raw_text]),
                self.remove_outlier(pitch),
                self.remove_outlier(energy),
                self.remove_outlier(df),
                mel_spectrogram.shape[1],
            )
        else:
            return (
                "|".join([basename, speaker, text, raw_text]),
                self.remove_outlier(pitch),
                self.remove_outlier(energy),
                mel_spectrogram.shape[1],
            )

    def get_alignment(self, tier):
        """
        TextGridの階層（tier）から音素レベルのアライメントを取得します。

        パラメータ
        ----------
        tier : tgt.core.IntervalTier
            音素レベルのアノテーションを含むIntervalTier。

        戻り値
        -------
        phones : list of str
            先頭および末尾の無音をトリミングした音素（ラベル）のリスト。
        durations : list of int
            各音素の継続時間（フレーム／ホップ数）のリスト。
        start_time : float
            最初の無音以外の音素の開始時刻（秒）。
        end_time : float
            最後の無音以外の音素の終了時刻（秒）。
        """
        sil_phones = ["sil", "sp", "spn"]

        phones = []
        durations = []
        start_time = 0
        end_time = 0
        end_idx = 0
        for t in tier._objects:
            s, e, p = t.start_time, t.end_time, t.text

            # Trim leading silences
            if phones == []:
                if p in sil_phones:
                    continue
                else:
                    start_time = s

            if p not in sil_phones:
                # For ordinary phones
                phones.append(p)
                end_time = e
                end_idx = len(phones)
            else:
                # For silent phones
                phones.append(p)

            # 音素の長さを計算（各音素が何フレーム分続くかを整数値で表現）
            durations.append(
                int(
                    np.round(e * self.sampling_rate / self.hop_length)
                    - np.round(s * self.sampling_rate / self.hop_length)
                )
            )

        # Trim tailing silences
        phones = phones[:end_idx]
        durations = durations[:end_idx]

        return phones, durations, start_time, end_time

    def remove_outlier(self, values):
        values = np.array(values)
        p25 = np.percentile(values, 25)
        p75 = np.percentile(values, 75)
        lower = p25 - 1.5 * (p75 - p25)
        upper = p75 + 1.5 * (p75 - p25)
        normal_indices = np.logical_and(values > lower, values < upper)

        return values[normal_indices]

    def normalize(self, in_dir, mean, std):
        max_value = np.finfo(np.float64).min
        min_value = np.finfo(np.float64).max
        for filename in os.listdir(in_dir):
            filename = os.path.join(in_dir, filename)
            values = (np.load(filename) - mean) / std
            np.save(filename, values)

            max_value = max(max_value, max(values))
            min_value = min(min_value, min(values))

        return min_value, max_value
