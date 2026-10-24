from seg2mri.data import N_CLASSES, SliceDataset, VolumeDataset


def test_slice_dataset_shapes(smoke_root):
    ds = SliceDataset(smoke_root, "train", augment=False)
    item = ds[0]
    assert item["image"].shape == (4, 32, 32)
    assert item["cond"].shape == (N_CLASSES, 32, 32)
    assert float(item["image"].min()) >= -1.0 and float(item["image"].max()) <= 1.0
    assert set(item["cond"].unique().tolist()) <= {0.0, 1.0}


def test_volume_dataset_shapes(smoke_root):
    ds = VolumeDataset(smoke_root, "test", augment=False)
    item = ds[0]
    assert item["image"].shape[0] == 4
    assert item["cond"].shape[0] == N_CLASSES
    assert item["image"].shape[1:] == item["cond"].shape[1:]


def test_splits_do_not_leak(smoke_root):
    train = set(SliceDataset(smoke_root, "train", augment=False).meta["splits"]["train"])
    test = set(SliceDataset(smoke_root, "test", augment=False).meta["splits"]["test"])
    assert train.isdisjoint(test)
