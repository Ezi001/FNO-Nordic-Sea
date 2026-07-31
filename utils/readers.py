import xarray as xr

def load_nordic_sea_splits(ocean, forcing):

    n = len(ocean)

    train_end = int(0.7 * n)
    eval_end = int(0.85 * n)

    return {
        "train": (
            ocean[:train_end],
            forcing[:train_end]
        ),
        "eval": (
            ocean[train_end:eval_end],
            forcing[train_end:eval_end]
        ),
        "test": (
            ocean[eval_end:],
            forcing[eval_end:]
        )
    }

def load_nordic_sea(ocean_path, forcing_path):
    ocean = xr.open_mfdataset(ocean_path, chunks={'time': 100})
    forcing = xr.open_mfdataset(forcing_path, chunks={'time': 100})

    return ocean, forcing


loaders = {
    "nordic": load_nordic_sea,
}
