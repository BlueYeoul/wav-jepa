import argparse
import os
import pathlib
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torchcodec.decoders import AudioDecoder
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
from wjepa.utils import init_audio_model, load_checkpoint
from wjepa.models.feature_extractor import compute_audio_output_length

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

# Alphabet for LibriSpeech
# 0: blank (standard for CTC)
# 1: SPACE
# 2: '
# 3-28: A-Z
ALPHABET = " '" + "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
CHAR_TO_INT = {c: i + 1 for i, c in enumerate(ALPHABET)}
INT_TO_CHAR = {i + 1: c for i, c in enumerate(ALPHABET)}
NUM_TOKENS = len(ALPHABET) + 1  # +1 for blank at index 0

class TextTokenizer:
    def encode(self, text):
        return torch.tensor([CHAR_TO_INT[c] for c in text.upper() if c in CHAR_TO_INT], dtype=torch.long)
    
    def decode(self, tokens):
        res = []
        for t in tokens:
            t = t.item()
            if t in INT_TO_CHAR:
                res.append(INT_TO_CHAR[t])
        return "".join(res)

    def decode_ctc(self, tokens):
        """Greedy CTC decoding."""
        res = []
        last_t = 0
        for t in tokens:
            t = t.item()
            if t != 0 and t != last_t:
                if t in INT_TO_CHAR:
                    res.append(INT_TO_CHAR[t])
            last_t = t
        return "".join(res)

# ---------------------------------------------------------------------------
# WER / CER
# ---------------------------------------------------------------------------

def _edit_distance(a: list, b: list) -> int:
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if a[i-1] == b[j-1] else 1 + min(prev, dp[j], dp[j-1])
            prev = temp
    return dp[n]

def compute_wer(ref: str, hyp: str) -> float:
    r, h = ref.upper().split(), hyp.upper().split()
    return _edit_distance(r, h) / max(len(r), 1)

def compute_cer(ref: str, hyp: str) -> float:
    r = list(ref.upper().replace(" ", ""))
    h = list(hyp.upper().replace(" ", ""))
    return _edit_distance(r, h) / max(len(r), 1)

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class LibriSpeechCTCDataset(Dataset):
    def __init__(self, root, split="dev-clean", sample_rate=16000, max_samples=None,
                 min_sec=0.0, max_sec=float("inf"), is_test=False):
        super().__init__()
        self.sample_rate = sample_rate
        self.tokenizer = TextTokenizer()
        self.samples = []

        if is_test:
            n = max_samples or 10
            print("[Test Mode] Generating synthetic data (10 sec)...")
            for _ in range(n):
                dummy_audio = torch.randn(1, sample_rate * 10)
                dummy_text = "HELLO WORLD THIS IS A TEST OF THE SPEECH RECOGNITION SYSTEM"
                self.samples.append((dummy_audio, dummy_text))
            return

        root_path = pathlib.Path(root) / split
        if not root_path.exists():
            raise FileNotFoundError(f"Path not found: {root_path}")

        use_dur_filter = (min_sec > 0.0) or (max_sec < float("inf"))
        dur_tag = f"{min_sec}~{max_sec}s" if use_dur_filter else "all durations"
        limit_tag = str(max_samples) if max_samples else "all"
        print(f"Scanning {root_path} [{dur_tag}, max={limit_tag}]...")
        trans_files = list(root_path.rglob("*.trans.txt"))

        for trans_file in trans_files:
            with open(trans_file, "r") as f:
                for line in f:
                    parts = line.strip().split(" ", 1)
                    if len(parts) < 2:
                        continue
                    file_id, text = parts
                    audio_file = trans_file.parent / f"{file_id}.flac"
                    if not audio_file.exists():
                        continue
                    # duration 체크 — 필터 있을 때만 메타데이터 읽음
                    if use_dur_filter:
                        duration = AudioDecoder(str(audio_file)).metadata.duration_seconds
                        if not (min_sec <= duration <= max_sec):
                            continue
                    self.samples.append((audio_file, text))
                    if max_samples and len(self.samples) >= max_samples:
                        break
            if max_samples and len(self.samples) >= max_samples:
                break

        print(f"Loaded {len(self.samples)} samples from {split}.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        audio_item, text = self.samples[idx]
        
        if isinstance(audio_item, torch.Tensor):
            waveform = audio_item
        else:
            waveform, sr = torchaudio.load(audio_item)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if sr != self.sample_rate:
                waveform = T.Resample(orig_freq=sr, new_freq=self.sample_rate)(waveform)
        
        # waveform: (1, T)
        tokens = self.tokenizer.encode(text)
        
        return {
            "audio": waveform,
            "tokens": tokens,
            "text": text
        }

def collate_fn(batch):
    audios = [b["audio"] for b in batch]
    tokens = [b["tokens"] for b in batch]
    texts = [b["text"] for b in batch]
    
    audio_lengths = torch.tensor([a.shape[1] for a in audios], dtype=torch.long)
    token_lengths = torch.tensor([len(t) for t in tokens], dtype=torch.long)
    
    max_audio_len = audio_lengths.max().item()
    max_token_len = token_lengths.max().item()
    
    padded_audios = []
    for a in audios:
        pad = max_audio_len - a.shape[1]
        padded_audios.append(F.pad(a, (0, pad)))
    
    padded_tokens = []
    for t in tokens:
        pad = max_token_len - len(t)
        padded_tokens.append(F.pad(t, (0, pad), value=0)) # 0 is blank/pad
        
    return {
        "audio": torch.stack(padded_audios), # (B, 1, T)
        "tokens": torch.stack(padded_tokens), # (B, L)
        "audio_lengths": audio_lengths,
        "token_lengths": token_lengths,
        "texts": texts
    }

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(target_encoder, ctc_head, dataloader, device, tokenizer):
    target_encoder.eval()
    ctc_head.eval()
    total_wer = total_cer = 0.0
    count = 0

    with torch.no_grad():
        for batch in dataloader:
            audio = batch["audio"].to(device)
            audio_lengths = batch["audio_lengths"]

            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                feats = target_encoder([audio], training_mode=False)[0]
                logits = ctc_head(feats)

            input_lengths = compute_audio_output_length(audio_lengths).clamp(max=feats.size(1))
            preds = torch.argmax(logits.float(), dim=-1)  # (B, T)

            for i, ref in enumerate(batch["texts"]):
                hyp = tokenizer.decode_ctc(preds[i, :input_lengths[i]])
                total_wer += compute_wer(ref, hyp)
                total_cer += compute_cer(ref, hyp)
                count += 1

    # 학습 모드 복원 (patch_embed는 eval 유지)
    target_encoder.train()
    target_encoder.backbone.patch_embed.eval()
    ctc_head.train()

    n = max(count, 1)
    return total_wer / n, total_cer / n

# ---------------------------------------------------------------------------
# Training Logic
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="./train/librispeech_pretrain/latest.pth.tar")
    parser.add_argument("--data_root", type=str, default="/data/LibriSpeech")
    parser.add_argument("--train_split", type=str, default="train-other-500")
    parser.add_argument("--test_split",  type=str, default="test-clean")
    parser.add_argument("--max_train_samples", type=int, default=-1, help="-1 = all")
    parser.add_argument("--max_test_samples",  type=int, default=-1, help="-1 = all")
    parser.add_argument("--min_sec", type=float, default=0.0)
    parser.add_argument("--max_sec", type=float, default=float("inf"))
    parser.add_argument("--test", action="store_true", help="Run with synthetic data")
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--model_name", type=str, default="audio_transformer_base")
    parser.add_argument("--save_dir", type=str, default="./train/ctc_finetune")
    parser.add_argument("--ckpt_freq", type=int, default=1, help="매 N epoch마다 e{epoch}.pth.tar 저장")
    parser.add_argument("--log_freq", type=int, default=50, help="매 N step마다 step 로그 출력")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.save_dir, exist_ok=True)
    latest_path = os.path.join(args.save_dir, "latest.pth.tar")

    # 1. Load Model
    # We need to initialize them all to use the helper load_checkpoint,
    # then we will delete the ones we don't need to free memory.
    encoder, predictor = init_audio_model(
        device=device,
        model_name=args.model_name,
        # Other params should ideally match config_base.yaml
        pred_depth=4,
        pred_embed_dim=384,
        n_output_distillation=4, # Match pre-train config to avoid weight mismatches
    )
    
    # Target encoder is the one we want (EMA)
    target_encoder, target_predictor = init_audio_model(
        device=device,
        model_name=args.model_name,
        pred_depth=4,
        pred_embed_dim=384,
        n_output_distillation=4, # Match pre-train config
    )

    if args.checkpoint and os.path.exists(args.checkpoint):
        # We only need target_encoder weights
        load_checkpoint(
            args.checkpoint, encoder, predictor, target_encoder, opt=None, scaler=None
        )
        print(f"Loaded EMA weights into target_encoder from {args.checkpoint}")
    else:
        if not args.test:
            print("[Warning] No checkpoint provided, training linear head on top of random encoder.")

    # Free unnecessary models to save memory
    del encoder
    del predictor
    del target_predictor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Freeze patch_embed (feature extractor) only
    target_encoder.backbone.patch_embed.eval()
    for p in target_encoder.backbone.patch_embed.parameters():
        p.requires_grad = False

    # 2. Linear CTC Head
    embed_dim = target_encoder.embed_dim
    ctc_head = nn.Linear(embed_dim, NUM_TOKENS).to(device)

    # 3. Data
    train_max = None if args.max_train_samples == -1 else args.max_train_samples
    test_max  = None if args.max_test_samples  == -1 else args.max_test_samples

    dataset = LibriSpeechCTCDataset(
        args.data_root, split=args.train_split,
        max_samples=train_max, min_sec=args.min_sec, max_sec=args.max_sec,
        is_test=args.test,
    )
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

    test_dataset = LibriSpeechCTCDataset(
        args.data_root, split=args.test_split,
        max_samples=test_max, is_test=False,
    )
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

    # 4. Optimizer & Loss — transformer blocks + ctc_head 함께 학습
    trainable_params = [p for p in target_encoder.parameters() if p.requires_grad] + list(ctc_head.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)
    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)

    ACCUM_STEPS = 8  # effective batch 128 / batch_size 16
    from wjepa.schedulers import WarmupCosineSchedule
    iterations_per_epoch = len(dataloader)
    optimizer_steps_per_epoch = max(1, iterations_per_epoch // ACCUM_STEPS)
    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=int(0.1 * args.epochs * optimizer_steps_per_epoch),
        start_lr=1e-4,
        ref_lr=args.lr,
        final_lr=5e-5,
        T_max=args.epochs * optimizer_steps_per_epoch
    )

    tokenizer = TextTokenizer()
    ustep = 0
    last_iter_ms = 0.0

    print(
        f"\nTraining start — {len(dataset)} train / {len(test_dataset)} test  "
        f"| {iterations_per_epoch} steps/epoch  "
        f"| effective batch {ACCUM_STEPS * 16}  "
        f"| {args.epochs} epochs\n"
    )

    # 5. Loop
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        step_loss  = 0.0
        optimizer.zero_grad()
        for step, batch in enumerate(dataloader):
            t0 = time.time()
            audio = batch["audio"].to(device)
            tokens = batch["tokens"].to(device)
            audio_lengths = batch["audio_lengths"]
            token_lengths = batch["token_lengths"]

            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                enc_outs = target_encoder([audio], training_mode=False)
                feats = enc_outs[0]
                logits = ctc_head(feats)  # (B, N, NUM_TOKENS)

            # CTC Loss expects (T, B, C) — float32 required
            log_probs = F.log_softmax(logits.float(), dim=-1).transpose(0, 1)
            input_lengths = compute_audio_output_length(audio_lengths).clamp(max=feats.size(1))
            loss = ctc_loss(log_probs, tokens, input_lengths, token_lengths)

            (loss / ACCUM_STEPS).backward()
            epoch_loss += loss.item()
            step_loss  += loss.item()
            last_iter_ms = (time.time() - t0) * 1000.0

            is_last = (step + 1 == len(dataloader))
            if (step + 1) % ACCUM_STEPS == 0 or is_last:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                ustep += 1

            # -- Step-level log --
            if (step + 1) % args.log_freq == 0 or is_last:
                mem_mb = (torch.cuda.max_memory_allocated() / 1024 ** 2
                          if torch.cuda.is_available() else 0.0)
                print(
                    f"[{epoch+1:4d}/{args.epochs}, step {step+1:5d}/{iterations_per_epoch},"
                    f" iter {ustep:6d}] "
                    f"loss: {step_loss / args.log_freq:.3f}  "
                    f"lr: {optimizer.param_groups[0]['lr']:.2e}  "
                    f"mem: {mem_mb:.0f}MB  iter: {last_iter_ms:.1f}ms"
                )
                step_loss = 0.0

        avg_loss = epoch_loss / len(dataloader)
        current_lr = optimizer.param_groups[0]["lr"]

        # -- Checkpoint --
        save_dict = {
            "target_encoder": target_encoder.state_dict(),
            "encoder":        target_encoder.state_dict(),  # load_checkpoint 호환
            "predictor":      {},                           # placeholder
            "ctc_head":       ctc_head.state_dict(),
            "opt":            optimizer.state_dict(),
            "scaler":         None,
            "epoch":          epoch + 1,
            "loss":           avg_loss,
            "batch_size":     16,
            "world_size":     1,
            "lr":             current_lr,
        }
        try:
            torch.save(save_dict, latest_path)
            if (epoch + 1) % args.ckpt_freq == 0:
                torch.save(save_dict, os.path.join(args.save_dir, f"e{epoch+1}.pth.tar"))
        except Exception as e:
            print(f"[warn] checkpoint save failed: {e}")

        # -- Log --
        if (epoch + 1) % 10 == 0:
            mem_mb = (torch.cuda.max_memory_allocated() / 1024 ** 2
                      if torch.cuda.is_available() else 0.0)
            with torch.no_grad():
                toks = torch.argmax(logits[0], dim=-1)
                hyp  = tokenizer.decode_ctc(toks)
                ref  = batch["texts"][0]
            print(
                f"[{epoch+1:4d}/{args.epochs}, iter {ustep:6d}] "
                f"loss: {avg_loss:.3f}  "
                f"lr: {current_lr:.2e}  "
                f"mem: {mem_mb:.0f}MB  iter: {last_iter_ms:.1f}ms"
            )
            print(f"  Ref: {ref}")
            print(f"  Hyp: {hyp}")

    # 6. Final evaluation on test split
    print(f"\n=== Evaluating on {args.test_split} ({len(test_dataset)} samples) ===")
    wer, cer = evaluate(target_encoder, ctc_head, test_dataloader, device, tokenizer)
    print(f"WER: {wer*100:.2f}%  CER: {cer*100:.2f}%")

if __name__ == "__main__":
    main()
