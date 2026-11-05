# Data

Downloaded data is **not committed**. Fetch it with:

```
python scripts/get_dataset.py
```

That pulls the **Middlebury MiddEval3** stereo benchmark at quarter
resolution (~45 MB) into `data/middlebury/MiddEval3/`:

```
trainingQ/<Scene>/
  im0.png          left image  (rectified)
  im1.png          right image
  disp0GT.pfm      dense float ground-truth disparity, inf where unknown
  mask0nocc.png    255 = non-occluded (the default evaluation region)
  calib.txt        intrinsics, baseline, and the scene's disparity range
```

15 training scenes with ground truth: Adirondack, ArtL, Jadeplant,
Motorcycle, MotorcycleE, Piano, PianoL, Pipes, Playroom, Playtable,
PlaytableP, Recycle, Shelves, Teddy, Vintage.

Quarter resolution (450x375 to 741x497) is deliberate -- it matches the "Q"
tables on the Middlebury leaderboard, and keeps a from-scratch NumPy matcher
at a second or two per pair instead of minutes.

Three scenes are deliberately awkward and worth looking at separately:
`ArtL`, `MotorcycleE` and `PianoL` are mis-exposed or differently lit between
the two views. Intensity-based costs (SAD, SSD) degrade there while the census
transform, which keeps only the ordering of neighbouring pixels, does not.

Each scene's `calib.txt` states its own `ndisp`, and the loader uses it:
Jadeplant needs 160 disparities, Teddy 64.

## Other sources

- `synthetic` -- generated in-process with exact ground truth, no download
  needed. See `bench/datasets/synthetic.py`.
- any folder with `im0`/`im1` (or `left`/`right`), e.g. the full-resolution
  pair in `notebooks/practice/`:
  `python -m bench run -d notebooks/practice --downscale 6`.

## What is committed here

Nothing but this file. `data/` is ignored wholesale (`data/*` with
`!data/README.md`), because everything in it is either downloaded by
`scripts/get_dataset.py` or uploaded through the app's Datasets tab. See
`docs/repository.md`.

## Citation

D. Scharstein, H. Hirschmuller, Y. Kitajima, G. Krathwohl, N. Nesic, X. Wang,
P. Westling. *High-resolution stereo datasets with subpixel-accurate ground
truth.* German Conference on Pattern Recognition (GCPR), 2014.
<https://vision.middlebury.edu/stereo/data/scenes2014/>
