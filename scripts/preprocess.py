"""
Name: Esther Zijerveld
Date: 4.8.2026
Location: University of Oslo, Informatics
"""
"""
Necessary imports
"""
import glob
import os

import argparse
import numpy as np
import pandas as pd

import xesmf as xe
import xarray as xr
import dask

# Creating directory for storing preprocessed data
os.makedirs("processed", exist_ok=True)

"""
Loading the data
Creating an argument parser function for parsing the start date and end date
"""

def parse_args() -> argparse.Namespace:
    """
    Parsing the start date and end date of the time series data
    """
    # Initialising parser
    parser = argparse.ArgumentParser()

    parser.add_argument("--start-date", type=str, default="1980-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default="1980-12-31", help="End date (YYYY-MM-DD)")

    return parser.parse_args()

# Accessing the parsed arguments
args = parse_args()

start_date = np.datetime64(args.start_date)
end_date = pd.Timestamp(args.end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

start_year = pd.Timestamp(start_date).year
end_year = pd.Timestamp(end_date).year
start_month = pd.Timestamp(start_date).month
end_month = pd.Timestamp(end_date).month


# Downloading the Era5 files
forcing_files = []

for year in range(start_year, end_year + 1):
    forcing_files.extend(glob.glob(f"data/ECMWF/fno_ERA5forcing_y{year}*.nc"))

if not forcing_files:
    raise FileNotFoundError(
        "No ERA5 forcing files found for the requested period "
        f"{start_date} to {end_date}."
    )

print("Files loaded:")
for f in forcing_files:
    print(f)

# Opening all the forcing files based on year
forcing = xr.open_mfdataset(forcing_files, 
                            combine="by_coords", 
                            chunks={'time': 100},
                            engine="netcdf4"
)

# Selecting only the specific time period of interest
forcing = forcing.sel(time=slice(start_date, end_date))
print("Shape: \n ", forcing["msl"].shape)

def load_nemo_variable(variable, start_date, end_date):
    """
    Function for 
    """
    files = []
    start_year = pd.Timestamp(start_date).year
    end_year = pd.Timestamp(end_date).year


    for year in range(start_year, end_year + 1):
        files.append(
            f"data/NEMO/NAA10KM_1h_{year}0101_{year}1231_{variable}.nc"
        )

    missing_files = [file_path for file_path in files if not os.path.isfile(file_path)]
    if missing_files:
        missing = "\n".join(f"  - {file_path}" for file_path in missing_files)
        raise FileNotFoundError(
            f"Missing NEMO {variable} file(s) for the requested period:\n{missing}"
        )

    ds = xr.open_mfdataset(files, 
                            combine="by_coords", 
                            chunks={"time_counter": 100},
                            engine="netcdf4",
    )

    ds = ds.rename({"time_counter": "time"})
    ds = ds.sel(time=slice(start_date, end_date)) # Selecting the correct time period

    return ds

# Only looking at sea surface height for now
nemo_ssh = load_nemo_variable("ssh", start_date, end_date)
print(nemo_ssh["ssh"].shape)

# nemo_ubar = load_nemo_variable(
#     "ubar",
#     args.start_year,
#     args.end_year
# )
# nemo_vbar = load_nemo_variable(
#     "vbar",
#     args.start_year,
#     args.end_year
# )

# Contains the ocean mask
bathy = xr.open_dataset("data/nordic_seas_domain_cfg.nc")


print("\n===== BATHYMETRY CHECK =====")
print("bathy dims:", bathy["top_level"].dims)
print("bathy shape:", bathy["top_level"].shape)
print("SSH dims:", ssh_6h.dims)
print("SSH shape:", ssh_6h.shape)

print("top_level unique:",
      np.unique(bathy["top_level"].values))

print("ocean points:",
      ocean_mask.sum().compute().item())

print("land points:",
      (~ocean_mask).sum().compute().item())

# print(nemo_ubar)
# print(nemo_ubar.coords)
# print(nemo_ubar.dims)


"""
Checking the longitude conventions
"""
print("\n===== COORDINATE CHECK =====")
print(f"ERA5 latitude:  {float(forcing.lat.min())} to {float(forcing.lat.max())}")
print(f"ERA5 longitude: {float(forcing.lon.min())} to {float(forcing.lon.max())}")
print(f"NEMO latitude:  {float(nemo_ssh.nav_lat.min())} to {float(nemo_ssh.nav_lat.max())}")
print(f"NEMO longitude: {float(nemo_ssh.nav_lon.min())} to {float(nemo_ssh.nav_lon.max())}")


"""
Crop ERA5 data to NEMO window
"""

buffer = 5.0

# Convert ERA5 longitude to -180..180 if necessary
# if float(forcing.lon.max()) > 180:
#     forcing = forcing.assign_coords(
#         lon=((forcing.lon + 180) % 360) - 180
#     ).sortby("lon")


# --------------------------------------------------
# Latitude slice
# --------------------------------------------------

lat_min = float(nemo_ssh.nav_lat.min()) - buffer
lat_max = float(nemo_ssh.nav_lat.max()) + buffer

if forcing.lat[0] > forcing.lat[-1]:
    lat_slice = slice(lat_max, lat_min)
else:
    lat_slice = slice(lat_min, lat_max)


# --------------------------------------------------
# Longitude slice
# --------------------------------------------------

lon_min = float(nemo_ssh.nav_lon.min()) - buffer
lon_max = float(nemo_ssh.nav_lon.max()) + buffer

if forcing.lon[0] > forcing.lon[-1]:
    lon_slice = slice(lon_max, lon_min)
else:
    lon_slice = slice(lon_min, lon_max)


# --------------------------------------------------
# Crop
# --------------------------------------------------

forcing_crop = forcing.sel(
    lat=lat_slice,
    lon=lon_slice,
)

print("\n===== ERA5 CROP =====")
print("lat:", forcing_crop.lat.values)
print("lon:", forcing_crop.lon.values)
# print("shape:", era5_crop.shape)
# print("dims:", era5_crop.dims)

if forcing_crop.sizes["lat"] == 0:
    raise ValueError(
        "ERA5 crop contains zero latitude points. "
        "Check latitude ordering and coordinate ranges."
    )

if forcing_crop.sizes["lon"] == 0:
    raise ValueError(
        "ERA5 crop contains zero longitude points. "
        "Check longitude convention and coordinate ranges."
    )


"""
Downsample to 6-hourly
Variations shorter than 6 hour is negligable
for a trajectory window of 6 month T is 182x4=728
"""
forcing_6h = forcing_crop.isel(time=slice(None, None, 2)) # downsample by 2

# Interpolating the time of the forcing to the 
ssh_6h = nemo_ssh["ssh"].interp(time=forcing_6h.time)
# ubar_6h = nemo_ubar["ubar"].isel(time=slice(None, None, 6))
# vbar_6h = nemo_vbar["vbar"].isel(time=slice(None, None, 6))
print("\n===== SSH NaN CHECK =====")

print(
    "Original NEMO SSH contains NaNs:",
    nemo_ssh["ssh"].isnull().any().compute().item()
)

print(
    "Interpolated SSH contains NaNs:",
    ssh_6h.isnull().any().compute().item()
)

print(
    "Number of NaNs in original SSH:",
    nemo_ssh["ssh"].isnull().sum().compute().item()
)

print(
    "Number of NaNs after time interpolation:",
    ssh_6h.isnull().sum().compute().item()
)



print(forcing_6h.coords)
assert np.array_equal(
    forcing_6h.time.values,
    ssh_6h.time.values
)
print(f"the new forcing time \n {forcing_6h.time}")
print(f"new time values for ocean {ssh_6h.time}")
# print(ubar_6h.time)
# print(vbar_6h.time)

# print(
#     (ssh_6h.time == ubar_6h.time).all()
# )
print(
    ((nemo_ssh["nav_lat"] >= 30) &
     (nemo_ssh["nav_lat"] <= 84)).sum().compute()
)
# --------------------------------------------------
# 1. Geographic latitude mask
# --------------------------------------------------

lat_mask = (
    (nemo_ssh["nav_lat"] >= 30) &
    (nemo_ssh["nav_lat"] <= 84)
)

ocean_mask = bathy["top_level"] > 0

combined_mask = ocean_mask & lat_mask

print("\n===== MASK DIAGNOSTIC =====")

print("SSH shape:", ssh_6h.shape)
print("Bathy shape:", bathy["top_level"].shape)

print("SSH y/x:", ssh_6h.sizes["y"], ssh_6h.sizes["x"])
print("Bathy y/x:", bathy["top_level"].sizes["y"], bathy["top_level"].sizes["x"])

print(
    "y coordinates equal:",
    np.array_equal(ssh_6h.y.values, bathy["top_level"].y.values)
)

print(
    "x coordinates equal:",
    np.array_equal(ssh_6h.x.values, bathy["top_level"].x.values)
)

print("top_level unique:",
      np.unique(bathy["top_level"].values))

print("Ocean points:",
      ocean_mask.sum().compute().item())

print("Land points:",
      (~ocean_mask).sum().compute().item())

print("Combined-mask ocean points:",
      combined_mask.sum().compute().item())

nan_mask = ssh_6h.isnull()

print("Total SSH NaNs:",
      nan_mask.sum().compute().item())

print("SSH NaNs over ocean:",
      (nan_mask & ocean_mask).sum().compute().item())

print("SSH NaNs over land:",
      (nan_mask & ~ocean_mask).sum().compute().item())

ssh_masked = ssh_6h.where(combined_mask, other=0)

print("NaNs after masking:",
      ssh_masked.isnull().sum().compute().item())

print("Nonzero values over land:",
      np.abs(ssh_masked.where(~combined_mask)).sum().compute().item())

print("===========================")

ssh_masked = ssh_6h.where(
    combined_mask,
    other=NaN
)
# --------------------------------------------------
# 6. Convert to (time, y, x, channel)
# --------------------------------------------------
state = (
    ssh_masked
    .expand_dims(channel=["ssh"])
    .transpose("time", "y", "x", "channel")
    .chunk({"time": 100})
)



# ubar_masked = ubar_6h.where(mask, other=0).assign_coords(
#     nav_lat=nemo_ssh["nav_lat"],
#     nav_lon=nemo_ssh["nav_lon"],
# )

# vbar_masked = vbar_6h.where(mask, other=0).assign_coords(
#     nav_lat=nemo_ssh["nav_lat"],
#     nav_lon=nemo_ssh["nav_lon"],
# )

print(ssh_masked.attrs)

print("ssh masked values: \n", state.shape)
print(ocean_mask.shape)
print(state.shape)
print(ocean_mask.dims)
print(state.dims)

"""
Interpolation from atmospheric data to sea surface height grid
Using the XESMF regridder for bilinear interpolation
To get them all on the ssh T grid, when we have the u and v velocities we also need to interpolate them to the T grid
"""
print("NEMO nav_lat:", nemo_ssh.nav_lat.dims, nemo_ssh.nav_lat.shape)
print("NEMO nav_lon:", nemo_ssh.nav_lon.dims, nemo_ssh.nav_lon.shape)

assert nemo_ssh.nav_lat.dims == ("y", "x")
assert nemo_ssh.nav_lon.dims == ("y", "x")

nemo_grid = xr.Dataset(
    {
        "lat": ssh_masked["nav_lat"],
        "lon": ssh_masked["nav_lon"],
    }
)

weight_file = "processed/era5_to_nemo_bilinear_weights.nc"


regridder_forcing = xe.Regridder(
    forcing_6h,
    nemo_grid,
    method="bilinear",
    filename=weight_file, # Where to save the weights
)
print(regridder_forcing)

# Regridding using bilinear interpolation
forcing_on_ssh = regridder_forcing(forcing_6h)



print("Interpolated forcing: \n", forcing_on_ssh)
print("Coordinates: \n", forcing_on_ssh.coords)
print("Values: \n", forcing_on_ssh["u10"].values)

#print(ssh_6h.time)


# Creating the forcing tensor
forcing_tensor = xr.concat(
    [
        forcing_on_ssh["u10"],
        forcing_on_ssh["v10"],
        forcing_on_ssh["msl"],
    ],
    dim="channel"
)

forcing_tensor = forcing_tensor.assign_coords(
    channel=["u10", "v10", "msl"]
)

forcing_tensor = forcing_tensor.rename("forcing")

forcing_tensor = forcing_tensor.transpose("time", "y", "x", "channel")

forcing_tensor = forcing_tensor.chunk(
    {
        "time": 100,
    }
)

print(forcing_tensor.sel(channel="msl").max().compute())
print(forcing_tensor.sel(channel="msl").min().compute())

print("\n===== GRID CHECK =====")

print("SSH:")
print("  dims:", ssh_masked.dims)
print("  shape:", ssh_masked.shape)
print("  nav_lat:", ssh_masked.nav_lat.shape)
print("  nav_lon:", ssh_masked.nav_lon.shape)

print("\nFORCING:")
print("  dims:", forcing_tensor.dims)
print("  shape:", forcing_tensor.shape)
print("  nav_lat:", forcing_tensor.nav_lat.shape)
print("  nav_lon:", forcing_tensor.nav_lon.shape)

print(forcing_on_ssh)
print(forcing_on_ssh.dims)

assert forcing_on_ssh.sizes["y"] == nemo_ssh.sizes["y"]
assert forcing_on_ssh.sizes["x"] == nemo_ssh.sizes["x"]

for var in ["u10", "v10", "msl"]:
    print(var, forcing_on_ssh[var].isnull().any().compute())

out_name = (
    f"{args.start_date}_{args.end_date}"
)

print("\nFORCING TENSOR")
print(forcing_tensor)

print("\nFORCING COORDS")
print(forcing_tensor.coords)

forcing_ds = xr.Dataset(
    {
        "forcing": forcing_tensor
    },
    coords={
        "nav_lat": nemo_ssh.nav_lat,
        "nav_lon": nemo_ssh.nav_lon,
    }
)

# ####################################
# Save processed variables
######################################
forcing_write = forcing_ds.to_netcdf(
    f"processed/forcing_{out_name}.nc",
    compute=False
)

state_write = state.to_netcdf(
    f"processed/state_{out_name}.nc",
    compute=False
)
mask_write = ocean_mask.to_netcdf(
    f"processed/mask_{out_name}.nc",
    compute=False
)

print("Saved variables to netcdf scripts.")

print(state)
print(forcing_tensor)
dask.compute(state_write, forcing_write, mask_write)


assert state.dims == ("time", "y", "x", "channel")
assert forcing_tensor.dims == ("time", "y", "x", "channel")
assert state.sizes["channel"] == 1
assert forcing_tensor.sizes["channel"] == 3
assert state.sizes["time"] == forcing_tensor.sizes["time"]
