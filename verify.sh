uv run verify.py \
        ./train/librispeech_pretrain/latest.pth.tar \
        --audio /data/LibriSpeech/test-other/1688/142285/1688-142285-0000.flac
# uv run check_fe.py ./train/librispeech_pretrain/latest.pth.tar   --audio /data/LibriSpeech/test-other/1688/142285/1688-142285-0000.flac