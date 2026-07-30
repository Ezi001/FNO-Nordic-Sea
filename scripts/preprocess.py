import xarray as xr
import os
os.makedirs("processed", exist_ok=True)

# Loading the data
forcing = xr.open_mfdataset("data/ECMWF/fno_ERA5forcing*.nc", chunks={'time': 100})
nemo_ssh = xr.open_mfdataset("data/*_ssh.nc", chunks={'time_counter': 100})
nemo_ubar = xr.open_mfdataset("data/*_ubar.nc", chunks={'time_counter':100})
nemo_vbar = xr.open_mfdataset("data/*_vbar.nc", chunks={'time_counter':100})
bathy = xr.open_dataset("data/nordic_seas_domain_cfg.nc")

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



# Interpolate ubar, vbar to ssh grid
import xesmf as xe

regridder_u = xe.Regridder(
    nemo_ubar,
    nemo_ssh,
    method="bilinear",
    reuse_weights=True
)

ubar_t = regridder_u(nemo_ubar["ubar"])

regridder_v = xe.Regridder(
    nemo_vbar,
    nemo_ssh,
    method="bilinear",
    reuse_weights=True
)

vbar_t = regridder_v(nemo_vbar["vbar"])



# Downsample to 6-hourly
#times_6h = forcing.time[::2]

ssh_6h = nemo_ssh["ssh"].isel(time_counter=slice(None, None, 6))
ubar_6h = ubar_t.isel(time_counter=slice(None, None, 6))
vbar_6h = vbar_t.isel(time_counter=slice(None, None, 6))
forcing_6h = forcing.isel(time=slice(None, None, 6))


# Crop Era5 data to NEMO window


buffer = 5  # degrees

era5_crop = forcing_6h.sel(
    lat=slice(lat_max + buffer, lat_min - buffer),  # ERA5 latitude decreases
    lon=slice(lon_min - buffer, lon_max + buffer)
)

mask = bathy["top_level"] == 1

# 1. Normalise (standardise) all variables
def normalize(da):
    mean = da.mean(skipna=True)
    std = da.std(skipna=True)
    max_val = da.max(skipna=True)
    min_val = da.min(skipna=True)

    normalized = (da - mean) / std

    return normalized, mean, std, max_val, min_val

ssh_norm, mean_ssh, std_ssh, max_ssh, min_ssh = normalize(ssh_6h.where(mask))
ubar_norm, mean_ubar, std_ubar, max_ubar , min_ubar = normalize(ubar_6h.where(mask))
vbar_norm, mean_vbar, std_vbar, max_vbar, min_vbar = normalize(vbar_6h.where(mask))

var_names = ["u10", "v10", "msl"]
mean = {v: era5_crop[v].mean(dim=("time", "lat", "lon"), skipna=True) for v in var_names}
std = {v: era5_crop[v].std(dim=("time", "lat", "lon"), skipna=True) for v in var_names}
maxs = {v: era5_crop[v].max(dim=("time", "lat", "lon"), skipna=True) for v in var_names}
mins = {v: era5_crop[v].min(dim=("time", "lat", "lon"), skipna=True) for v in var_names}

forcing_norm = xr.Dataset({
    v: (era5_crop[v] - mean[v]) / std[v]
    for v in var_names
})

normalization = xr.Dataset({
    "mean_ssh": mean_ssh,
    "std_ssh": std_ssh,
    "maks_ssh": max_ssh,
    "min_ssh": min_ssh,
    "mean_ubar": mean_ubar,
    "std_ubar": std_ubar,
    "max_ubar": max_ubar,
    "min_ubar": min_ubar,
    "mean_vbar": mean_vbar,
    "std_vbar": std_vbar,
    "max_vbar": max_vbar,
    "min_vbar": min_vbar,
    **{
        f"mean_{v}": mean[v]
        for v in var_names
    },
    **{
        f"std_{v}": std[v]
        for v in var_names
    },
    **{
        f"max_{v}": maxs[v]
        for v in var_names
    },

    **{
        f"min_{v}": mins[v]
        for v in var_names
    },
})



# 2. Land mask for the ocean variables
ssh_masked = ssh_norm.where(mask, other=0).assign_coords(
    nav_lat=nemo_ssh["nav_lat"],
    nav_lon=nemo_ssh["nav_lon"],
)

ubar_masked = ubar_norm.where(mask, other=0).assign_coords(
    nav_lat=nemo_ssh["nav_lat"],
    nav_lon=nemo_ssh["nav_lon"],
)

vbar_masked = vbar_norm.where(mask, other=0).assign_coords(
    nav_lat=nemo_ssh["nav_lat"],
    nav_lon=nemo_ssh["nav_lon"],
)

target_lat = nemo_ssh["nav_lat"]
target_lon = nemo_ssh["nav_lon"]

# Interpolation from atmospheric data to sea surface height grid
forcing_on_ssh = forcing_norm.interp(
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


state = xr.concat(
    [
        ssh_masked,
        ubar_masked,
        vbar_masked,
    ],
    dim="channel"
) # (T, H, W, 3)


state = state.assign_coords(
    channel=["ssh", "ubar", "vbar"]
)



# Save processed variables
state.to_netcdf("processed/state.nc") # (time, y, x, 3)
forcing_tensor.to_netcdf("processed/forcing.nc") # (time_y, x, 3)
mask.to_netcdf("processed/ocean_mask.nc")
normalization.to_netcdf(
    "processed/normalization_stats.nc"
)