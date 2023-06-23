# Data Preparing

The directory structure of the dataset is as follows:

```bash
.
└── dataset
    └──sun_glint
        └── data
        │    ├── images
        │    ├── masks (binary masks, (H,W,1))
        │    ├── train.txt
        │    ├── val.txt
        │    └── test.txt
        └── sun_glint_classes_weights.npy
```

Example images and masks can be found in `./Sun_glint_detection/dataset/sun_glint/examples/`