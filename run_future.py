import matplotlib.pyplot as plt
import gcsfs
import glob
import jax
import numpy as np
import pickle
import xarray
import sys
from netCDF4 import Dataset
import os
import pandas as pd

from dinosaur import horizontal_interpolation
from dinosaur import spherical_harmonic
from dinosaur import xarray_utils
import neuralgcm

gcs = gcsfs.GCSFileSystem(token='anon')

model_name = 'neural_gcm_dynamic_forcing_deterministic_2_8_deg.pkl'

with gcs.open(f'gs://gresearch/neuralgcm/04_30_2024/{model_name}', 'rb') as f:
  ckpt = pickle.load(f)

model = neuralgcm.PressureLevelModel.from_checkpoint(ckpt)

demo_start_time = '2000-04-01'
demo_end_time = '2001-05-31'
data_inner_steps = int(24)  # process every 24 hours

ens_num = int(1000)
loop_number = 71

era5_cache_dir = './era5_cache/'
cached_files = sorted(glob.glob(os.path.join(era5_cache_dir, 'era5_*.nc')))

if cached_files:
    print('loading era5 from local daily cache files')
    eval_era5_base = xarray.open_mfdataset(cached_files, combine='by_coords')
else:
    print('reading era5 from GCS')
    era5_path = 'gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3'
    full_era5 = xarray.open_zarr(gcs.get_mapper(era5_path), chunks=None)

    era5_grid = spherical_harmonic.Grid(
        latitude_nodes=full_era5.sizes['latitude'],
        longitude_nodes=full_era5.sizes['longitude'],
        latitude_spacing=xarray_utils.infer_latitude_spacing(full_era5.latitude),
        longitude_offset=xarray_utils.infer_longitude_offset(full_era5.longitude),
    )
    regridder = horizontal_interpolation.ConservativeRegridder(
        era5_grid, model.data_coords.horizontal, skipna=True
    )

    # apply shift lazily, then compute and save one day at a time
    era5_shifted = (
        full_era5
        [model.input_variables + model.forcing_variables]
        .pipe(
            xarray_utils.selective_temporal_shift,
            variables=model.forcing_variables,
            time_shift='24 hours',
        )
    )
    time_steps = era5_shifted.sel(time=slice(demo_start_time, demo_end_time, data_inner_steps)).time.values

    os.makedirs(era5_cache_dir, exist_ok=True)
    print('saving era5 daily files')
    for t in time_steps:
        date_str = str(t)[:10]
        ds_t = era5_shifted.sel(time=[t]).compute()
        ds_t = xarray_utils.regrid(ds_t, regridder)
        ds_t = xarray_utils.fill_nan_with_nearest(ds_t)
        ds_t.to_netcdf(os.path.join(era5_cache_dir, f'era5_{date_str}.nc'))

    cached_files = sorted(glob.glob(os.path.join(era5_cache_dir, 'era5_*.nc')))
    eval_era5_base = xarray.open_mfdataset(cached_files, combine='by_coords')

inner_steps = int(24)  # save model outputs once every 24 hours
outer_steps = 6 * 24 // inner_steps
timedelta = np.timedelta64(1, 'h') * inner_steps
times = (np.arange(outer_steps) * inner_steps)

# ================================================================
# read pamip profiles once (future)
# ================================================================
print('reading pamip future profiles')
dirname = './future_pamip/'
filename = 'daily_forcing_2_8_deg_future_pamip.nc'
f = Dataset(dirname + filename, 'r')
sic_pamip = f.variables['sic'][:,:,:].data
sst_pamip = f.variables['sst'][:,:,:].data
f.close()

os.makedirs('./future', exist_ok=True)

for MMM in range(ens_num):

    # load only the needed time window; SST/SIC are overwritten with PAMIP data anyway
    eval_era5 = eval_era5_base.isel(time=slice(0, outer_steps)).compute()

    print('initialization...')

    print('=====before=====')
    print('sic', np.nanmean(eval_era5.sea_ice_cover.values))
    print('sst', np.nanmean(eval_era5.sea_surface_temperature.values))

    eval_era5.sea_ice_cover.values[:,:,:] = sic_pamip[:outer_steps,:,:].copy()
    eval_era5.sea_surface_temperature.values[:,:,:] = sst_pamip[:outer_steps,:,:].copy()

    print('=====after=====')
    print('sic', np.nanmean(eval_era5.sea_ice_cover.values))
    print('sst', np.nanmean(eval_era5.sea_surface_temperature.values))

    # initialize model state
    inputs = model.inputs_from_xarray(eval_era5.isel(time=0))

    # add perturbation to temperature field
    print('add pertubation to temperature field: ', (MMM+1)*1.e-4)
    inputs['temperature'][-2,111,11] = inputs['temperature'][-2,111,11] + (MMM+1)*1.e-4

    input_forcings = model.forcings_from_xarray(eval_era5.isel(time=0))
    rng_key = jax.random.key(MMM)  # optional for deterministic models
    print(rng_key)
    key, subkey = jax.random.split(rng_key)
    print(subkey)
    initial_state = model.encode(inputs, input_forcings, subkey)

    filename = './future/' + 'future_member' + str(MMM+1).zfill(3) + '.nc'
    if os.path.exists(filename):
        os.remove(filename)
    f_out = None

    for NTT in range(loop_number):

        print('making forecast...')
        print('loop number:', NTT)

        print('update sst and sic')
        eval_era5.sea_ice_cover.values[:,:,:] = sic_pamip[NTT*outer_steps:NTT*outer_steps+outer_steps,:,:].copy()
        eval_era5.sea_surface_temperature.values[:,:,:] = sst_pamip[NTT*outer_steps:NTT*outer_steps+outer_steps,:,:].copy()
        eval_era5 = eval_era5.assign_coords(time=eval_era5.time + pd.Timedelta(days=6))

        all_forcings = model.forcings_from_xarray(eval_era5)

        # make forecast
        final_state, predictions = model.unroll(
            initial_state,
            all_forcings,
            steps=outer_steps,
            timedelta=timedelta,
            start_with_input=True,
        )
        predictions_ds = model.data_to_xarray(predictions, times=times)

        combined_ds = xarray.concat([predictions_ds], 'model')
        combined_ds.coords['model'] = ['NeuralGCM']

        lev = np.asarray(combined_ds.level).copy()
        nz = len(lev)
        lat = np.asarray(combined_ds.latitude).copy()
        ny = len(lat)
        lon = np.asarray(combined_ds.longitude).copy()
        nx = len(lon)
        t_start = NTT * outer_steps

        if NTT == 0:
            # compression: zlib + byte-shuffle; chunk = one (model,time) slice
            comp = dict(zlib=True, complevel=4, shuffle=True)
            ck5 = (1, 1, nz, nx, ny)
            ck3 = (1, nx, ny)
            units = {'v': 'm s-1', 'u': 'm s-1', 'q': 'kg kg-1', 'z': 'm2 s-2',
                     't': 'K', 'sst': 'K', 'sic': '1',
                     'lev': 'hPa', 'lat': 'degrees_north', 'lon': 'degrees_east'}

            f_out = Dataset(filename, 'w', format='NETCDF4')
            f_out.createDimension('model', 1)
            f_out.createDimension('time', None)  # unlimited: grows each loop
            f_out.createDimension('lev', nz)
            f_out.createDimension('lat', ny)
            f_out.createDimension('lon', nx)
            # CF time coordinate: daily steps from demo_start_time
            tc = f_out.createVariable('time', 'i4', ('time',))
            tc.units = 'days since ' + demo_start_time + ' 00:00:00'
            tc.calendar = 'standard'
            tc.long_name = 'time'
            for c in ('lev', 'lat', 'lon'):
                cv = f_out.createVariable(c, 'f4', (c,))
                cv.units = units[c]
            for v in ('v', 'u', 'q', 'z', 't'):
                vv = f_out.createVariable(v, 'f4', ('model', 'time', 'lev', 'lon', 'lat'),
                                          chunksizes=ck5, **comp)
                vv.units = units[v]
            for v in ('sst', 'sic'):
                vv = f_out.createVariable(v, 'f4', ('time', 'lon', 'lat'),
                                          chunksizes=ck3, **comp)
                vv.units = units[v]
            f_out['lev'][:] = lev
            f_out['lat'][:] = lat
            f_out['lon'][:] = lon

        f_out['time'][t_start:t_start+outer_steps] = np.arange(t_start, t_start+outer_steps)
        f_out['v'][:, t_start:t_start+outer_steps, :, :, :] = np.asarray(combined_ds.v_component_of_wind)
        f_out['u'][:, t_start:t_start+outer_steps, :, :, :] = np.asarray(combined_ds.u_component_of_wind)
        f_out['q'][:, t_start:t_start+outer_steps, :, :, :] = np.asarray(combined_ds.specific_humidity)
        f_out['z'][:, t_start:t_start+outer_steps, :, :, :] = np.asarray(combined_ds.geopotential)
        f_out['t'][:, t_start:t_start+outer_steps, :, :, :] = np.asarray(combined_ds.temperature)
        f_out['sst'][t_start:t_start+outer_steps, :, :] = np.asarray(eval_era5.sea_surface_temperature)
        f_out['sic'][t_start:t_start+outer_steps, :, :] = np.asarray(eval_era5.sea_ice_cover)

        print(np.nanmean(np.asarray(combined_ds.temperature)))

        # update initial condition
        initial_state = final_state

    f_out.close()
    print('saved', filename)
