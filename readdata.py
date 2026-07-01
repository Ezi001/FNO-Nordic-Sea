# File for visualising data sets

import xarray as xr
import matplotlib.pyplot as plt
import ncplot.xarray

# Load the entire file
data = xr.open_dataset('data\\fno_ERA5forcing_y1980m01.nc')
data_ssh = xr.open_dataset('data\\NAA10KM_1h_19800101_19801231_ssh.nc') # Sea surface height
data_ubar = xr.open_dataset('data\\NAA10KM_1h_19800101_19801231_ubar.nc') # U
data_vbar = xr.open_dataset('data\\NAA10KM_1h_19800101_19801231_vbar.nc') # V
data_bath = xr.open_dataset('data\\nordic_seas_domain_cfg.nc')


#print(type(data))
#print(type(data["msl"]))

# Explore the data
#print(data)  # Shows all variables, dimensions, and metadata
#print(data_ssh)
##print(data_ubar)
#print(data_vbar)
print(data_vbar)


# Access a specific variable
u10 = data['u10']  # Replace 'ssh' with your variable name
#print(u10.values)  # Get the numpy array

for i in range(0, 190):
    data_vbar["vbar"].isel(time_counter=i).plot()
    plt.show()

data_vbar["vbar"].isel(time_counter=8700).plot()
plt.show()

print(data.lat.shape, data.lon.shape)
#print(data_ssh.lat.shape, data_ssh.lon.shape)
#print(data_ubar.lat.shape, data_ubar.lon.shape)
#print(data_vbar.lat.shape, data_vbar.lon.shape)
#print(data_bath.lat.shape, data_bath.lon.shape)

