# Datasets

Images are **not committed** (see `.gitignore`). Each of us downloads them locally into the folders below.
Only clean images are used; all noise is added by our own code.

```
data/
├── set12/      # 12 grayscale images, 256x256   (Phase 1 sanity checks)
├── kodak24/    # 24 colour images, 768x512      (main benchmark, converted to gray on load)
└── bsd68/      # 68 grayscale images, ~481x321  (final confirmation run)
```

Where to get them (check each link still works):

| Dataset | Source |
|---|---|
| Set12 | `testsets/Set12` in the DnCNN repository (github.com/cszn/DnCNN) |
| BSD68 | `testsets/BSD68` in the DnCNN repository (github.com/cszn/DnCNN) |
| Kodak24 | `testsets/Kodak24` in the FFDNet repository (github.com/cszn/FFDNet), or the original Kodak set `kodim01.png` ... `kodim24.png` from r0k.us/graphics/kodak |

Supported file types: `.png .jpg .jpeg .bmp .tif .tiff`.

Until the real data is in place, the notebook falls back to a few scikit-image sample images so you can still run everything.
