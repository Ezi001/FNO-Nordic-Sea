"""
Necessary imports
"""
import xarray as xr
import os
import xesmf as xe

os.makedirs("processed", exist_ok=True)

"""
Loading the data
"""
forcing = xr.open_mfdataset("data/ECMWF/fno_ERA5forcing*.nc", chunks={'time': 100})
nemo_ssh = xr.open_mfdataset("data/*_ssh.nc", chunks={'time_counter': 100}).rename({"time_counter": "time"})
nemo_ubar = xr.open_mfdataset("data/*_ubar.nc", chunks={'time_counter':100}).rename({"time_counter": "time"})
nemo_vbar = xr.open_mfdataset("data/*_vbar.nc", chunks={'time_counter':100}).rename({"time_counter": "time"})
bathy = xr.open_dataset("data/nordic_seas_domain_cfg.nc")

"""
Checking the longitude conventions
"""
lat_min = float(nemo_ssh['ssh'].nav_lat.min())
lat_max = float(nemo_ssh['ssh'].nav_lat.max())

lon_min = float(nemo_ssh['ssh'].nav_lon.min())
lon_max = float(nemo_ssh['ssh'].nav_lon.max())

print(forcing.lon.min().item(), forcing.lon.max().item())
print(float(lon_min), float(lon_max))

if forcing.lon.max() > 180 and lon_min < 0:
    raise ValueError(
        "ERA5 uses 0-360 longitude while NEMO uses -180-180."
    )

"""
Downsample to 6-hourly

"""
ssh_6h = nemo_ssh["ssh"].isel(time=slice(None, None, 6))
ubar_6h = nemo_ubar["ubar"].isel(time=slice(None, None, 6))
vbar_6h = nemo_vbar["vbar"].isel(time=slice(None, None, 6))

forcing_6h = forcing.interp(time=ssh_6h.time)

"""
Interpolate ubar, vbar to ssh T-grid
"""

regridder_u = xe.Regridder(
    ubar_6h,
    ssh_6h,
    method="bilinear",
    reuse_weights=True
)

ubar_t = regridder_u(ubar_6h)

regridder_v = xe.Regridder(
    vbar_6h,
    ssh_6h,
    method="bilinear",
    reuse_weights=True
)

vbar_t = regridder_v(vbar_6h)


"""
Crop Era5 data to NEMO window
"""

buffer = 5  # degrees

era5_crop = forcing_6h.sel(
    lat=slice(lat_max + buffer, lat_min - buffer),  # ERA5 latitude decreases
    lon=slice(lon_min - buffer, lon_max + buffer)
)

mask = bathy["top_level"] == 1


"""
2. Land mask for the ocean variables
"""
ssh_masked = ssh_6h.where(mask, other=0).assign_coords(
    nav_lat=nemo_ssh["nav_lat"],
    nav_lon=nemo_ssh["nav_lon"],
)

ubar_masked = ubar_t.where(mask, other=0).assign_coords(
    nav_lat=nemo_ssh["nav_lat"],
    nav_lon=nemo_ssh["nav_lon"],
)

vbar_masked = vbar_t.where(mask, other=0).assign_coords(
    nav_lat=nemo_ssh["nav_lat"],
    nav_lon=nemo_ssh["nav_lon"],
)

target_lat = nemo_ssh["nav_lat"]
target_lon = nemo_ssh["nav_lon"]

# Interpolation from atmospheric data to sea surface height grid
forcing_on_ssh = era5_crop.interp(
    lat=target_lat,
    lon=target_lon,
)

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
forcing_tensor = forcing_tensor.transpose("time", "y", "x", "channel")

state = xr.concat(
    [
        ssh_masked,
        ubar_masked,
        vbar_masked,
    ],
    dim="channel"
) # (channel, time, y, x)


state = state.assign_coords(
    channel=["ssh", "ubar", "vbar"]
)
state = state.transpose("time", "y", "x", "channel")

print(state.dims)
print(state.shape)

print(forcing_tensor.dims)
print(forcing_tensor.shape)

# Save processed variables
state.to_netcdf("processed/state.nc") # (time, y, x, 3)
forcing_tensor.to_netcdf("processed/forcing.nc") # (time, y, x, 3)
mask.to_netcdf("processed/ocean_mask.nc")


assert state.dims == ("time", "y", "x", "channel")
assert forcing_tensor.dims == ("time", "y", "x", "channel")
assert state.sizes["channel"] == 3
assert forcing_tensor.sizes["channel"] == 3
assert state.sizes["time"] == forcing_tensor.sizes["time"]
