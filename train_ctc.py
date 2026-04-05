import argparse
import os
import pathlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
from wjepa.utils import init_audio_model, load_checkpoint

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
# Dataset
# ---------------------------------------------------------------------------

class LibriSpeechCTCDataset(Dataset):
    def __init__(self, root, split="dev-clean", sample_rate=16000, max_samples=10, is_test=False):
        super().__init__()
        self.sample_rate = sample_rate
        self.tokenizer = TextTokenizer()
        self.samples = []
        
        if is_test:
            print("[Test Mode] Generating synthetic data...")
            for i in range(max_samples):
                dummy_audio = torch.randn(1, 16000 * 2) # 2 seconds
                dummy_text = "HELLO WORLD TEST"
                self.samples.append((dummy_audio, dummy_text))
            return

        root_path = pathlib.Path(root) / split
        if not root_path.exists():
            raise FileNotFoundError(f"Path not found: {root_path}")

        print(f"Scanning {root_path} for transcripts...")
        trans_files = list(root_path.rglob("*.trans.txt"))
        
        for trans_file in trans_files:
            with open(trans_file, "r") as f:
                for line in f:
                    parts = line.strip().split(" ", 1)
                    if len(parts) < 2:
                        continue
                    file_id, text = parts
                    # Find audio file
                    audio_file = trans_file.parent / f"{file_id}.flac"
                    if audio_file.exists():
                        self.samples.append((audio_file, text))
                        if len(self.samples) >= max_samples:
                            break
                if len(self.samples) >= max_samples:
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
# Training Logic
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="./train/librispeech_pretrain/latest.pth.tar", help="Path to W-JEPA checkpoint")
    parser.add_argument("--data_root", type=str, default="/data/LibriSpeech", help="LibriSpeech root")
    parser.add_argument("--test", action="store_true", help="Run with synthetic data")
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--model_name", type=str, default="audio_transformer_base")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

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

    # Freeze target_encoder
    # target_encoder.eval()
    for p in target_encoder.backbone.patch_embed.parameters():
        p.requires_grad = False

    # 2. Linear CTC Head
    embed_dim = target_encoder.embed_dim
    ctc_head = nn.Linear(embed_dim, NUM_TOKENS).to(device)

    # 3. Data
    dataset = LibriSpeechCTCDataset(args.data_root, max_samples=10, is_test=args.test)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

    # 4. Optimizer & Loss
    optimizer = torch.optim.AdamW(ctc_head.parameters(), lr=args.lr)
    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)

    from wjepa.schedulers import WarmupCosineSchedule
    iterations_per_epoch = len(dataloader)
    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=int(0.1 * args.epochs * iterations_per_epoch), # 10% warmup
        start_lr=1e-4,
        ref_lr=args.lr,
        final_lr=5e-5,
        T_max=args.epochs * iterations_per_epoch
    )

    tokenizer = TextTokenizer()

    # 5. Loop
    print(f"Starting linear probing (CTC) with LR range {args.lr} -> 5e-5...")
    for epoch in range(args.epochs):
        epoch_loss = 0
        for batch in dataloader:
            audio = batch["audio"].to(device)
            tokens = batch["tokens"].to(device)
            audio_lengths = batch["audio_lengths"]
            token_lengths = batch["token_lengths"]

            with torch.no_grad():
                # MultiSeqWrapper expects a list of tensors
                # training_mode=False returns last norm (B, N, D)
                enc_outs = target_encoder([audio], training_mode=False)
                # enc_outs is a list [ (B, N, D) ]
                feats = enc_outs[0]

            # Linear head
            logits = ctc_head(feats) # (B, N, NUM_TOKENS)
            
            # CTC Loss expects (T, B, C)
            log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
            
            # Compute output lengths after CNN (stride=320)
            input_lengths = torch.full((audio.size(0),), feats.size(1), dtype=torch.long)

            loss = ctc_loss(log_probs, tokens, input_lengths, token_lengths)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch+1}/{args.epochs}, Loss: {epoch_loss/len(dataloader):.4f}, LR: {current_lr:.6f}")
            # Show a sample decode
            with torch.no_grad():
                sample_logits = logits[0:1] # (1, N, C)
                sample_tokens = torch.argmax(sample_logits, dim=-1)[0]
                decoded = tokenizer.decode_ctc(sample_tokens)
                print(f"  Ref: {batch['texts'][0]}")
                print(f"  Hyp: {decoded}")

    print("Verification complete.")

if __name__ == "__main__":
    main()
