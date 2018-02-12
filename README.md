# Dataset

Get the GOPRO dataset, and save it to `images/train`

# Installation

```
virtualenv venv -p python3
. venv/bin/activate
pip install -r requirements.txt
```

# Training

```
python train.py --n_images=512 --batch_size=16
```

Use `python train.py --help` for all options

# Testing

```
python test.py
```

Use `python test.py --help` for all options
