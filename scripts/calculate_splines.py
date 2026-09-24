import xarray as xr
import zarr
import numpy as np

from utils.splines_torch import quintic_spline_batch


DATA_DIR = "C:/Users/esthe/Documents/UiO/FNO-Nordic-Sea/data"

trajectory_length = 50
stride = 49


def calculate_splines(
    data,
    trajectory_length,
    stride,
    output_path,
):
    n_windows = len(
        range(
            0,
            len(data) - trajectory_length + 1,
            stride,
        )
    )

    print(f"Number of trajectories: {n_windows}")

    # --------------------------------------------------
    # First trajectory determines output shape
    # --------------------------------------------------

    trajectory = data.isel(
        time=slice(0, trajectory_length)
    ).values

    trajectory = trajectory[None, ...]

    coeffs, starts = quintic_spline_batch(
        trajectory,
        time=None,
        batch_size=1,
        num_point=3,
        device="cpu",
    )

    print("Coefficient shape:", coeffs.shape)
    print("Start shape:", starts.shape)

    # --------------------------------------------------
    # Create Zarr group
    # --------------------------------------------------

    root = zarr.open_group(
        output_path,
        mode="w",
    )

    coefficients = root.create_array(
        "coefficients",
        shape=(n_windows,) + coeffs.shape[1:],
        dtype=coeffs.dtype,
        chunks=(1,) + coeffs.shape[1:],
    )

    spline_starts = root.create_array(
        "starts",
        shape=(n_windows,) + starts.shape[1:],
        dtype=starts.dtype,
        chunks=(1,) + starts.shape[1:],
    )

    # --------------------------------------------------
    # Save trajectory start/end times
    # --------------------------------------------------

    t_start = root.create_array(
        "t_start",
        shape=(n_windows,),
        dtype="datetime64[ns]",
        chunks=(1,),
    )

    t_end = root.create_array(
        "t_end",
        shape=(n_windows,),
        dtype="datetime64[ns]",
        chunks=(1,),
    )

    # --------------------------------------------------
    # Loop over trajectories
    # --------------------------------------------------

    for i, start in enumerate(
        range(
            0,
            len(data) - trajectory_length + 1,
            stride,
        )
    ):
        trajectory = data.isel(
            time=slice(
                start,
                start + trajectory_length,
            )
        ).values

        trajectory = trajectory[None, ...]

        coeffs, starts = quintic_spline_batch(
            trajectory,
            time=None,
            batch_size=1,
            num_point=3,
            device="cpu",
        )

        coefficients[i] = coeffs[0]
        spline_starts[i] = starts[0]

        # Physical time of first and last trajectory point
        t_start[i] = data.time.isel(
            time=start
        ).values

        t_end[i] = data.time.isel(
            time=start + trajectory_length - 1
        ).values

        if i % 10 == 0:
            print(
                f"Saved {i + 1}/{n_windows}"
            )

    print(f"Finished: {output_path}")


# ---------------------------------------------------------
# Load ocean state
# ---------------------------------------------------------

ds = xr.open_zarr(
    f"{DATA_DIR}/nemo_ocean.zarr"
)

ssh_train = ds["ssh"].sel(
    time=slice(
        "2020-06-01",
        "2022-12-31",
    )
)

ssh_heldout = ds["ssh"].sel(
    time=slice(
        "2023-01-01",
        "2024-12-31",
    )
)

print("Train:", ssh_train.shape)
print("Heldout:", ssh_heldout.shape)


# ---------------------------------------------------------
# Calculate splines
# ---------------------------------------------------------

calculate_splines(
    ssh_train,
    trajectory_length=trajectory_length,
    stride=stride,
    output_path=f"{DATA_DIR}/splines_train.zarr",
)

calculate_splines(
    ssh_heldout,
    trajectory_length=trajectory_length,
    stride=stride,
    output_path=f"{DATA_DIR}/splines_heldout.zarr",
)
