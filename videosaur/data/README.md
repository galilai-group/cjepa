


# Dataset Creation Scripts

### MOVi

Preliminaries:

```
poetry install -E tensorflow
```

**MOVi-C**

```
python save_movi.py --level c --split train --maxcount 32 --only-video <root_data_dir>/movi_c
python save_movi.py --level c --split validation --maxcount 32 <root_data_dir>/movi_c
```

**MOVi-E**

```
python save_movi.py --level e --split train --maxcount 32 --only-video <root_data_dir>/movi_e
python save_movi.py --level e --split validation --maxcount 32 <root_data_dir>/movi_e
```

### COCO

Preliminaries:

```
poetry install -E coco
```

```
python save_coco.py --split train --maxcount 128 --only-images --out-path <root_data_dir>/coco
python save_coco.py --split validation --maxcount 128 --out-path <root_data_dir>/coco
python save_coco.py --split unlabeled --maxcount 256 --original-image-format --out-path <root_data_dir>/coco
```

Note that the script downloads and extracts the raw COCO dataset to `--download-dir`.

### YouTube-VIS 2021

Preliminaries:

```
poetry install -E coco
```

```
python save_ytvis2021.py --split train --maxcount 32 --only-videos --resize --out-path <root_data_dir>/ytvis2021_resized
python save_ytvis2021.py  --split validation --maxcount 10 --resize --out-path <root_data_dir>/ytvis2021_resized
```

### YouTube-VIS 2019

Preliminaries:
```
poetry install -E tensorflow
```

```
python save_ytvis2019.py --split train --maxcount 32 --only-videos --resize --out-path <root_data_dir>/ytvis2019_resized
python save_ytvis2019.py  --split validation --maxcount 10 --resize --out-path <root_data_dir>/ytvis2019_resized
```

Note that we expect that raw files are downloaded in corresponding `ytvis2019_raw` and `ytvis2021_raw` folders.
