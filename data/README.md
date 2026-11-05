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
  needed. One image is warped into the other, so occlusion is approximated
  rather than modelled. See `bench/datasets/synthetic.py`.
- `synth:<environment>` -- ray-traced 3-D scenes, also in-process and also
  exact, but with *physical* occlusion: both views are traced independently,
  so the non-occluded mask is computed rather than guessed. See
  `bench/synth/` and the README section on generating training images.
- any folder with `im0`/`im1` (or `left`/`right`), e.g. the full-resolution
  pair in `practice/`: `python -m bench run -d practice --downscale 6`.

## Generated datasets

`python -m bench synth generate --out data/synth/<name>` writes one folder per
pair in exactly the MiddEval3 layout above, so generated data loads through
the same code path as a downloaded scene. By default the pairs are divided
into disjoint subsets:

```
data/synth/<name>/
  train/00000/ ...    80%
  val/00008/   ...    10%
  test/00009/  ...    10%
```

Counts are exact, and every pair has its own seed and its own scene, so no
geometry is shared between splits. `--splits ''` writes one flat directory
instead, and any split loads on its own:

```
python -m bench run -d data/synth/<name>/test -a opencv_sgbm
```

Alongside the samples it writes:

```
manifest.json    the rules, the seeds, and the measured statistics per pair
rules.json       the rule set on its own, enough to regenerate bit for bit
README.md        what the set contains and how it was made
```

`data/synth/` is not committed -- regenerate it from `rules.json`.

## Citation

D. Scharstein, H. Hirschmuller, Y. Kitajima, G. Krathwohl, N. Nesic, X. Wang,
P. Westling. *High-resolution stereo datasets with subpixel-accurate ground
truth.* German Conference on Pattern Recognition (GCPR), 2014.
<https://vision.middlebury.edu/stereo/data/scenes2014/>
