import torch
import numpy as np
from torch.utils.data import Dataset
import pathlib

import torchaudio
import torchaudio.transforms as T

class AudioDataset(Dataset):
    """
    Audio dataset class for variable-length processing.
    """
    def __init__(self, file_list, sample_rate=16000, min_sec=1.5, max_sec=20.0):
        super().__init__()
        self.sample_rate = sample_rate
        self.min_samples = int(min_sec * sample_rate)
        self.max_samples = int(max_sec * sample_rate)
        # 파일 리스트를 직접 받음 (Factory에서 필터링된 결과)
        self.audio_files = list(file_list)

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        # print(f"[Debug] Fetching index {idx} - File: {self.audio_files[idx]}")  # Debug log
        while True:
            audio_file = self.audio_files[idx]
            audio = self._load_audio(audio_file)
            if audio is not None:
                break
            # Drop된 경우 다른 랜덤 데이터로 재시도
            idx = torch.randint(0, len(self.audio_files), (1,)).item()
            if idx == 0 and len(self.audio_files) == 1: # 무한루프 방지
                break

        actual_len = len(audio)
        return {
            "audio": audio,
            "seq_len_sec": actual_len / self.sample_rate,
            "seq_len": actual_len
        }

    def _load_audio(self, path):
        waveform, sr = torchaudio.load(path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != self.sample_rate:
            waveform = T.Resample(orig_freq=sr, new_freq=self.sample_rate)(waveform)
        waveform = waveform.squeeze(0)

        audio_len = len(waveform)
        if audio_len < self.min_samples:
            return None
        if audio_len > self.max_samples:
            max_start = audio_len - self.max_samples
            start_idx = torch.randint(0, max_start + 1, (1,)).item()
            waveform = waveform[start_idx : start_idx + self.max_samples]
        return waveform

class LibriSpeechDatasetFactory:
    """
    LibriSpeech specific factory to handle modes (100, 360, 500, 960).
    """
    @staticmethod
    def create(root_path, mode=500, sample_rate=16000, min_sec=1.5, max_sec=20.0):
        # 1. 하드코딩된 모드 정의 및 기본값 처리
        valid_modes = {100, 360, 500, 960}
        if mode not in valid_modes:
            print(f"[Warning] Mode {mode} is invalid. Defaulting to 500.")
            mode = 500
        
        root = pathlib.Path(root_path)
        if not root.exists():
            raise FileNotFoundError(f"Root path {root_path} does not exist.")

        print(f"[Info] Loading LibriSpeech Mode: {mode}")

        # 2. 모드 숫자가 포함된 폴더를 찾고, 그 안의 오디오 파일들을 수집
        target_files = []
        exts = [".wav", ".flac", ".mp3"]
        
        # root 하위의 모든 디렉토리를 탐색
        for subdir in root.iterdir():
            if subdir.is_dir() and str(mode) in subdir.name:
                print(f"[Info] Scanning directory: {subdir.name}")
                # 해당 폴더 내부에서 오디오 파일 찾기
                for ext in exts:
                    target_files.extend(list(subdir.rglob(f"*{ext}")))

        if not target_files:
            print(f"[Warning] No audio files found for mode {mode} in {root_path}")
            # 빈 리스트라도 넘겨서 에러 방지 (AudioDataset의 __getitem__에서 대응 필요)
            return AudioDataset([], sample_rate=sample_rate, min_sec=min_sec, max_sec=max_sec)

        print(f"[Info] Found {len(target_files)} audio files.")
        
        # 3. 수집된 파일 리스트로 Dataset 반환
        return AudioDataset(
            file_list=target_files, 
            sample_rate=sample_rate, 
            min_sec=min_sec, 
            max_sec=max_sec
        )

# --- 사용 예시 ---
# if __name__ == "__main__":
#     # 실제 경로로 수정하여 테스트하세요.
#     LIBRI_ROOT = "/data/LibriSpeech"
    
#     try:
#         # Mode 500 (Default) 생성 시도
#         dataset_500 = LibriSpeechDatasetFactory.create(LIBRI_ROOT, mode=500)
#         print(f"Dataset 500 Size: {len(dataset_500)}")

#         if len(dataset_500) > 0:
#             sample = dataset_500[0]
#             print(f"Sample Audio Shape: {sample['audio'].shape}")

#         # Mode 100 생성 시도
#         dataset_100 = LibriSpeechDatasetFactory.create(LIBRI_ROOT, mode=100)
#         print(f"Dataset 100 Size: {len(dataset_100)}")

#     except Exception as e:
#         print(f"Error: {e}")



def collate_fn(batch):
    """
    배치 단위 패딩 및 Attention Mask 생성
    """
    audio_list = [item["audio"] for item in batch]
    lengths = torch.tensor([item["seq_len"] for item in batch], dtype=torch.long)
    max_len = lengths.max().item()

    # 배치 내 가장 긴 오디오에 맞춰 우측(오른쪽) Zero Padding
    audio_padded = [
        torch.nn.functional.pad(a, (0, max_len - len(a)), value=0.0)
        for a in audio_list
    ]
    audio_padded = torch.stack(audio_padded)

    return {
        "audio": audio_padded,         # (B, T)
        "lengths": lengths             # (B,)
    }
