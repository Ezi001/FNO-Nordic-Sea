import xarray as xr

zarr_path_ocean = "data/nemo_ocean.zarr"
zarr_path_forcing = "data/nemo_forcing.zarr"


ds_ocean = xr.open_zarr(zarr_path)
ds_forcing = xr.open_zarr(zarr_path)

print(ds_forcing.data_vars)
print(ds_forcing["time"].values)

start_date = np.datetime64("2020-06-01")
end_date = pd.Timestamp("2024-12-31") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)


forcing = ds_forcing.sel(time=slice(start_date, end_date))
ocean = ds_ocean.sel(time=slice(start_date, end_date)) # Selecting the correct time period
