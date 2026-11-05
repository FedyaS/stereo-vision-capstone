# Pretrained model weights

Empty by design -- weights are not committed. Fetch them with:

```
python scripts/get_models.py
```

That downloads ONNX exports of **RAFT-Stereo** (~70 MB total) from the
`AXERA-TECH/RAFT-stereo` mirror on Hugging Face. Without them the `learned_*`
algorithms report themselves as unavailable and the rest of the benchmark runs
normally.
