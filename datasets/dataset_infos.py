
import collections

BASE_DATA_DIR = "/home/user/DISK/data/Jing/data/"
DatasetDescriptor = collections.namedtuple(
    'DatasetDescriptor',
    [
        'splits_to_sizes',  # Splits of the dataset into training, val and test.
        'num_classes',  # Number of semantic classes, including the
                        # background class (if exists). For example, there
                        # are 13 foreground classes + 1 background class in
                        # the MICCAI BTCV dataset. Thus, we set
                        # num_classes=14.
        'channel', # The image channel. Most of the time,  medical image channel would be 1.
        'ignore_label',  # Ignore label value.
        'height', # raw data height
        'width', # raw data width
        'train', # training parameters
        'HU_window',
    ])


_ISBI_CHAOS_INFORMATION_CT = DatasetDescriptor(
    splits_to_sizes={
        'train': 2302,
        'val': 572,
        'test': 3533
    },
    num_classes=2, # liver and background
    channel=1,
    ignore_label=255,
    height=512,
    width=512,
    HU_window=[-125, 275],
    train={
        "train_crop_size": [256, 256],
        "pre_crop_size": [None, None],
    }
)


_ISBI_CHAOS_INFORMATION_MR_T2 = DatasetDescriptor(
    splits_to_sizes={
        'train': 501,
        'val': 122,
        'test': 645,
    },
    num_classes=5, # liver, right kidney, left kidney, spleen and background
    channel=1,
    ignore_label=255,
    height=256,
    width=256,
    HU_window=None,
    train={
        "train_crop_size": [256, 256],
        "pre_crop_size": [None, None],
    }
)


_ISBI_CHAOS_INFORMATION_MR_T1 = DatasetDescriptor(
    splits_to_sizes={
        'train': 521,
        'val': 126,
        'test': 653,
    },
    num_classes=5, # liver, right kidney, left kidney, spleen and background
    channel=1,
    ignore_label=255,
    height=256,
    width=256,
    HU_window=None,
    train={
        "train_crop_size": [256, 256],
        "pre_crop_size": [None, None],
    }
)
# height, width = 256, 288, 320, 400

_MICCAI_ABDOMINAL_INFORMATION = DatasetDescriptor(
    splits_to_sizes={
        'train': 3017,
        'val': 762,
        'test': 2387
    },
    num_classes=14,
    channel=1,
    ignore_label=255,
    height=512,
    width=512,
    HU_window=[-125, 275],
    train={
        "train_crop_size": [256, 256],
        "pre_crop_size": [460, 460],
    }
)
