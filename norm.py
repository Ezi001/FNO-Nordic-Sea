# 1. Normalise (standardise) all variables
def normalize(da):
    mean = da.mean(skipna=True)
    std = da.std(skipna=True)
    max_val = da.max(skipna=True)
    min_val = da.min(skipna=True)

    normalized = (da - mean) / std

    return normalized, mean, std, max_val, min_val

ssh_norm, mean_ssh, std_ssh, max_ssh, min_ssh = normalize(ssh_6h.where(mask))
ubar_norm, mean_ubar, std_ubar, max_ubar , min_ubar = normalize(ubar_t.where(mask))
vbar_norm, mean_vbar, std_vbar, max_vbar, min_vbar = normalize(vbar_t.where(mask))

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