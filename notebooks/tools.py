import os
import glob
import xdem
import geoutils as gu
import numpy as np
import datetime
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from IPython.display import HTML
import psutil
import multiprocessing as mp
import concurrent
import rioxarray
import xarray as xr
from sklearn import linear_model
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import ExpSineSquared
from sklearn.gaussian_process.kernels import RationalQuadratic
from sklearn.gaussian_process.kernels import WhiteKernel
import pandas as pd

###########
"""
From ragmac_xdem/data/raw/convert_dates.py
@author: brunbarf

Modified by @friedrichknuth
"""

def convert_date_time_to_decimal_date(date_time):
    """
    This function converts a date and a time to a decimal date value
    Inputs:
    - date_time: datetime object
    
    Outputs:
    - decimal_date_float: float
    
    """
    year_start   = datetime.date(date_time.year, 1, 1).toordinal()
    year_end     = datetime.date(date_time.year+1, 1, 1).toordinal()
    days_in_year = year_end - year_start
    decimal_date = date_time.year + float(date_time.toordinal() - year_start) / days_in_year
    return decimal_date
    
def convert_decimal_date_to_date_time(decimal_date):
    """
    This function converts a decimal date and a date and time
    Inputs:
    - decimal_date: float
    
    Outputs:
    - date_time: datetime object
    - date_time_string: formated string from the datetime object
    """
    year       = int(np.floor(decimal_date))
    year_start = datetime.date(year,1,1).toordinal()
    year_end     = datetime.date(year+1,1,1).toordinal()
    days_in_year = year_end - year_start
    days_since_year_begin = (decimal_date - year)*days_in_year
    date_time = datetime.date.fromordinal(int(np.floor(year_start + days_since_year_begin)))
    date_time_string=date_time.strftime('%Y-%m-%d')
    return date_time, date_time_string

###########  
"""
Modified from https://github.com/dshean/pygeotools/blob/master/pygeotools/lib/malib.py#L999
@author: dshean
"""
def calcperc(b, perc=(2.0,98.0)):
    """Calculate values at specified percentiles
    """
    b = checkma(b)
    if b.count() > 0:
        #low = scoreatpercentile(b.compressed(), perc[0])
        #high = scoreatpercentile(b.compressed(), perc[1])
        low = np.percentile(b.compressed(), perc[0])
        high = np.percentile(b.compressed(), perc[1])
    else:
        low = 0
        high = 0
    return low, high

def calcperc_sym(b, perc=(2.0,98.0)):
    """
    Get symmetrical percentile values
    Useful for determining clim centered on 0 for difference maps
    """
    clim = np.max(np.abs(calcperc(b, perc)))
    return -clim, clim

def checkma(a, fix=False):
    #isinstance(a, np.ma.MaskedArray)
    if np.ma.is_masked(a):
        out=a
    else:
        out=np.ma.array(a)
    #Fix invalid values
    #Note: this is not necessarily desirable for writing
    if fix:
        #Note: this fails for datetime arrays! Treated as objects.
        #Note: datetime ma returns '?' for fill value
        from datetime import datetime
        if isinstance(a[0], datetime):
            print("Input array appears to be datetime.  Skipping fix")
        else:
            out=np.ma.fix_invalid(out, copy=False)
    return out

def mad(a, axis=None, c=1.4826, return_med=False):
    """Compute normalized median absolute difference
   
    Can also return median array, as this can be expensive, and often we want both med and nmad
    Note: 1.4826 = 1/0.6745
    """
    a = checkma(a)
    #return np.ma.median(np.fabs(a - np.ma.median(a))) / c
    if a.count() > 0:
        if axis is None:
            med = fast_median(a)
            out = fast_median(np.fabs(a - med)) * c
        else:
            med = np.ma.median(a, axis=axis)
            #The expand_dims is necessary for broadcasting
            out = np.ma.median(np.ma.fabs(a - np.expand_dims(med, axis=axis)), axis=axis) * c
    else:
        out = np.ma.masked
    if return_med:
        out = (out, med)
    return out

def do_robust_linreg(arg):
    date_list_o, y, model = arg
    y_idx = ~(np.ma.getmaskarray(y))
    #newaxis is necessary b/c ransac expects 2D input array
    x = date_list_o[y_idx].data[:,np.newaxis]
    y = y[y_idx].data
    return robust_linreg(x, y, model)

def robust_linreg(x, y, model='theilsen'):
    from sklearn import linear_model
    slope = None
    intercept = None
    if model == 'linear':
        m = linear_model.LinearRegression()
        m.fit(x, y)
        slope = m.coef_
        intercept = m.intercept_
    elif model == 'ransac':
        m = linear_model.RANSACRegressor()
        m.fit(x, y)
        slope = m.estimator_.coef_
        intercept = m.estimator_.intercept_
        #inlier_mask = ransac.inlier_mask_
        #outlier_mask = np.logical_not(inlier_mask)
    elif model == 'theilsen':
        m = linear_model.TheilSenRegressor()
        m.fit(x, y)
        slope = m.coef_
        intercept = m.intercept_
    #xi = np.arange(x.min(), x.max())[:,np.newaxis]
    #yi = model.predict(xi) 
    #ax.plot(xi, yi)
    return(slope[0], intercept)

def ma_linreg(ma_stack, dt_list, n_thresh=2, model='linear', dt_stack_ptp=None, min_dt_ptp=None, smooth=False, \
        rsq=False, conf_test=False, parallel=True, n_cpu=None, remove_outliers=False):
    """Compute per-pixel linear regression for stack object
    """
    #Need to check type of input dt_list
    #For now assume it is Python datetime objects 
#     from pygeotools.lib import timelib
    date_list_o = np.ma.array(matplotlib.dates.date2num(dt_list))
    date_list_o.set_fill_value(0.0)

    #ma_stack = ma_stack[:,398:399,372:373]
    #dt_stack_ptp = dt_stack_ptp[398:399,372:373]

    #Only compute trend where we have n_thresh unmasked values in time
    #Create valid pixel count
    count = np.ma.masked_equal(ma_stack.count(axis=0), 0).astype(np.uint16).data
    print("Excluding pixels with count < %i" % n_thresh)
    valid_mask_2D = (count >= n_thresh)

    #Only compute trend where the time spread (ptp is max - min) is large
    if dt_stack_ptp is not None:
        if min_dt_ptp is None:
            #Calculate from datestack ptp
            max_dt_ptp = calcperc(dt_stack_ptp, (4, 96))[1]
            #Calculate from list of available dates
            #max_dt_ptp = np.ptp(calcperc(date_list_o, (4, 96)))
            min_dt_ptp = 0.10 * max_dt_ptp
        print("Excluding pixels with dt range < %0.2f days" % min_dt_ptp) 
        valid_mask_2D = valid_mask_2D & (dt_stack_ptp >= min_dt_ptp).filled(False)

    #Extract 1D time series for all valid pixel locations
    y_orig = ma_stack[:, valid_mask_2D]
    #Extract mask and invert: True where data is available
    valid_mask = ~(np.ma.getmaskarray(y_orig))
    valid_sample_count = np.inf

    if y_orig.count() == 0:
        print("No valid samples remain after count and min_dt_ptp filters")
        slope = None
        intercept = None
        detrended_std = None
    else:
        #Create empty (masked) output grids with original dimensions
        slope = np.ma.masked_all_like(ma_stack[0])
        intercept = np.ma.masked_all_like(ma_stack[0])
        detrended_std = np.ma.masked_all_like(ma_stack[0])

        #While loop here is to iteratively remove outliers, if desired
        #Maximum number of iterations
        max_n = 3
        n = 1
        while(y_orig.count() < valid_sample_count and n <= max_n):
            print(n)
            valid_pixel_count = np.sum(valid_mask_2D)
            valid_sample_count = y_orig.count()
            print("%i valid pixels with up to %i timestamps: %i total valid samples" % \
                    (valid_pixel_count, ma_stack.shape[0], valid_sample_count))
            if model == 'theilsen' or model == 'ransac':
                #Create empty arrays for slope and intercept results 
                m = np.ma.masked_all(y_orig.shape[1])
                b = np.ma.masked_all(y_orig.shape[1])
                if parallel:
                    import multiprocessing as mp
                    if n_cpu is None:
                        n_cpu = mp.cpu_count() - 1
                        #n_cpu = psutil.cpu_count(logical=True)
                    n_cpu = int(n_cpu)
                    print("Running in parallel with %i processes" % n_cpu)
                    pool = mp.Pool(processes=n_cpu)
                    results = pool.map(do_robust_linreg, [(date_list_o, y_orig[:,n], model) for n in range(y_orig.shape[1])])

                    results = np.array(results)
                    m = results[:,0]
                    b = results[:,1]
                else:
                    for n in range(y_orig.shape[1]):
                        print('%i of %i px' % (n, y_orig.shape[1]))
                        y = y_orig[:,n]
                        m[n], b[n] = do_robust_linreg([date_list_o, y, model])
            else:
                #if model == 'linear':
                #Remove masks, fills with fill_value
                y = y_orig.data
                #Independent variable is time ordinal
                x = date_list_o
                x_mean = x.mean()
                x = x.data
                #Prepare matrices
                X = np.c_[x, np.ones_like(x)]
                a = np.swapaxes(np.dot(X.T, (X[None, :, :] * valid_mask.T[:, :, None])), 0, 1)
                b = np.dot(X.T, (valid_mask*y))
            
                #Solve for slope/intercept
                print("Solving for trend")
                r = np.linalg.solve(a, b.T)
                #Reshape to original dimensions
                m = r[:,0]
                b = r[:,1]

            print("Computing residuals")
            #Compute model fit values for each valid timestamp
            y_fit = m*np.ma.array(date_list_o.data[:,None]*valid_mask, mask=y_orig.mask) + b 
            #Compute residuals
            resid = y_orig - y_fit
            #Compute detrended std
            #resid_std = resid.std(axis=0)
            resid_std = mad(resid, axis=0)

            if remove_outliers and n < max_n:
                print("Removing residual outliers > 3-sigma")
                outlier_sigma = 3.0
                #Mask any residuals outside valid range
                valid_mask = valid_mask & (np.abs(resid) < (resid_std * outlier_sigma)).filled(False)
                #Extract new valid samples
                y_orig = np.ma.array(y_orig, mask=~valid_mask)
                #Update valid mask
                valid_count = (y_orig.count(axis=0) >= n_thresh)
                y_orig = y_orig[:, valid_count]
                valid_mask_2D[valid_mask_2D] = valid_count
                #Extract 1D time series for all valid pixel locations
                #Extract mask and invert: True where data is available
                valid_mask = ~(np.ma.getmaskarray(y_orig))
                #remove_outliers = False
            else:
                break
            n += 1

        #Fill in the valid indices
        slope[valid_mask_2D] = m 
        intercept[valid_mask_2D] = b
        detrended_std[valid_mask_2D] = resid_std

        #Smooth the result
        if smooth:
            size = 5
            print("Smoothing output with %i px gaussian filter" % size)
            from pygeotools.lib import filtlib
            #Gaussian filter
            #slope = filtlib.gauss_fltr_astropy(slope, size=size)
            #intercept = filtlib.gauss_fltr_astropy(intercept, size=size)
            #Median filter
            slope = filtlib.rolling_fltr(slope, size=size, circular=False)
            intercept = filtlib.rolling_fltr(intercept, size=size, circular=False)

        if rsq:
            rsquared = np.ma.masked_all_like(ma_stack[0])
            SStot = np.sum((y_orig - y_orig.mean(axis=0))**2, axis=0).data
            SSres = np.sum(resid**2, axis=0).data
            r2 = 1 - (SSres/SStot)
            rsquared[valid_mask_2D] = r2

        if conf_test:
            count = y_orig.count(axis=0)
            SE = np.sqrt(SSres/(count - 2)/np.sum((x - x_mean)**2, axis=0))
            T0 = r[:,0]/SE
            alpha = 0.05
            ta = np.zeros_like(r2)
            from scipy.stats import t
            for c in np.unique(count):
                t1 = abs(t.ppf(alpha/2.0,c-2))
                ta[(count == c)] = t1
            sig = np.logical_and((T0 > -ta), (T0 < ta))
            sigmask = np.zeros_like(valid_mask_2D, dtype=bool)
            sigmask[valid_mask_2D] = ~sig
            #SSerr = SStot - SSres
            #F0 = SSres/(SSerr/(count - 2))
            #from scipy.stats import f
            #    f.cdf(sig, 1, c-2)
            slope = np.ma.array(slope, mask=~sigmask)
            intercept = np.ma.array(intercept, mask=~sigmask)
            detrended_std = np.ma.array(detrended_std, mask=~sigmask)
            rsquared = np.ma.array(rsquared, mask=~sigmask)
        
        #slope is in units of m/day since x is ordinal date
        slope *= 365.25

    return slope, intercept, detrended_std

########### 
"""
@author: friedrichknuth
"""

###########  Wrangling functions
def stack_raster_arrays(raster_files_list, parse_time_stamps=True):
    arrays = []
    dt_list = []

    for i in raster_files_list[:]:
        src = gu.georaster.Raster(i)
        masked_array = src.data

        arrays.append(masked_array)

        if parse_time_stamps:
            date_time = float(os.path.basename(i).split('_')[1])
            dt_list.append(date_time)

    ma_stack = np.ma.vstack(arrays)
    if parse_time_stamps:
        dt_list = [convert_decimal_date_to_date_time(i)[0] for i in dt_list]

        return ma_stack, dt_list
    
    else:
        return ma_stack, None

def xr_read_tif(tif_file_path, 
                chunks=1000, 
                masked=True):
    """
    Reads in single or multi-band GeoTIFF as chunked dask array for lazy io.
    Parameters
    ----------
    GeoTIFF_file_path : str
    Returns
    -------
    ds : xarray.Dataset
        Includes rioxarray extension to xarray.Dataset
    """

    da = rioxarray.open_rasterio(tif_file_path, chunks=chunks, masked=True)

    # Extract bands and assign as variables in xr.Dataset()
    ds = xr.Dataset()
    for i, v in enumerate(da.band):
        da_tmp = da.sel(band=v)
        da_tmp.name = "band" + str(i + 1)

        ds[da_tmp.name] = da_tmp

    # Delete empty band coordinates.
    # Need to preserve spatial_ref coordinate, even though it appears empty.
    # See spatial_ref attributes under ds.coords.variables used by rioxarray extension.
    del ds.coords["band"]

    # Preserve top-level attributes and extract single value from value iterables e.g. (1,) --> 1
    ds.attrs = da.attrs
    for key, value in ds.attrs.items():
        try:
            if len(value) == 1:
                ds.attrs[key] = value[0]
        except TypeError:
            pass

    return ds

def mask_low_count_pixels(ma_stack, n_thresh = 3):
    count = np.ma.masked_equal(ma_stack.count(axis=0), 0).astype(np.uint16).data
    valid_mask_2D = (count >= n_thresh)
    valid_data = ma_stack[:, valid_mask_2D]
    return valid_data, valid_mask_2D

###########  Plotting functions
def plot_array_gallery(array_3d,
                       titles_list = None,
                       figsize = (10,15),
                       vmin = None,
                       vmax = None,
                       cmap='viridis'):
    
    if not vmin:
        vmin = np.nanmin(array_3d)+50
    if not vmax:
        vmax = np.nanmax(array_3d)-50
    
    rows, columns = get_row_column(len(array_3d))
    fig = plt.figure(figsize=(10,15))

    for i in range(rows*columns):
        try:
            array = array_3d[i]
            ax = plt.subplot(rows, columns, i + 1, aspect='auto')
            ax.imshow(array,interpolation='none',cmap=cmap, vmin=vmin,vmax=vmax)
            ax.set_xticks(())
            ax.set_yticks(())
            if titles_list:
                ax.set_title(titles_list[i])
        except:
            pass
    plt.tight_layout()

def plot_time_series_gallery(x_values,
                             y_values,
                             predictions_df_list=None,
                             std_df_list=None,
                             x_ticks_off=False,
                             y_ticks_off=True,
                             sharex = True,
                             figsize=(10,10),
                             legend=True,
                             linestyle='-',
                             legend_labels = ["Observations",]):
        
    rows, columns = get_row_column(len(x_values))
    
    fig = plt.figure(figsize=figsize)
    axes = []
    for i in range(rows*columns):
        try:
            x, y = x_values[i], y_values[i]
            ax = plt.subplot(rows, columns, i + 1, aspect='auto')
            ax.plot(x, y, marker='o',c='b', label=legend_labels[0])
            if x_ticks_off:
                ax.set_xticks(())
            if y_ticks_off:
                ax.set_yticks(())
            axes.append(ax)
        except:
            pass
    if not isinstance(predictions_df_list, type(None)):
        for idx, df in enumerate(predictions_df_list):
            try:
                std_df = std_df_list[idx]
            except:
                std_df = None
            
            for i, series in df.iteritems():
                ax = axes[i]
                try:
                    series.plot(ax=ax,c='C'+str(idx+1),label= legend_labels[idx+1])
                except:
                    series.plot(ax=ax,c='C'+str(idx+1),label= 'Observations')
                if not isinstance(std_df, type(None)):
                    x = series.index.values
                    y = series.values
                    std_prediction = std_df[i].values
                    ax.fill_between(x,
                                    y - 1.96 * std_prediction,
                                    y + 1.96 * std_prediction,
                                    alpha=0.2,
                                    label=legend_labels[idx+1]+'_95_%conf',
                                    color='C'+str(idx+1))
    
    if legend:
        axes[0].legend()
    if sharex:
        for ax in axes[:-columns]:
            ax.set_xticks(())
    plt.tight_layout()
    
def plot_timelapse(array, 
                   figsize=(10,10),
                   points= None,
                   titles_list = None,
                   frame_rate=200,
                   vmin = None,
                   vmax = None,
                   alpha = None):
    '''
    array with shape (time, x, y)
    '''
    if not vmin:
        vmin = np.nanmin(array)+50
    if not vmax:
        vmax = np.nanmax(array)-50
        
    fig, ax = plt.subplots(figsize=(10,10))
    im = ax.imshow(array[0,:,:],interpolation='none',alpha = alpha,vmin=vmin,vmax=vmax)
    if points:
        p, = ax.plot(points[0],points[1],marker='o',color='b',linestyle='none')
    plt.close()
    
    def vid_init():
        im.set_data(array[0,:,:])
        if points:
            p.set_data(points[0],points[1])
    def vid_animate(i):
        im.set_data(array[i,:,:])
        if points:
            p.set_data(points[0],points[1])
        if titles_list:
            ax.set_title(titles_list[i])

    anim = animation.FuncAnimation(fig, 
                                   vid_animate, 
                                   init_func=vid_init, 
                                   frames=array.shape[0],
                                   interval=frame_rate)
    return HTML(anim.to_html5_video())

def plot_count_std(count_nmad_ma_stack,
                   count_vmin = 1,
                   count_vmax = 50,
                   count_cmap = 'gnuplot',
                   std_vmin = 0,
                   std_vmax = 20,
                   std_cmap = 'cividis',
                   points = None,
                   alpha = None,
                   ticks_off = True,
                   ):
    
    fig,axes = plt.subplots(1,2,figsize=(15,10))
    
    ax = axes[0]
    cmap = plt.cm.get_cmap(count_cmap, count_vmax)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    
    plt.colorbar(ax.imshow(count_nmad_ma_stack[0],
                           vmin=count_vmin,
                           vmax=count_vmax,
                           interpolation='none',
                           cmap=cmap,
                           alpha=alpha),
                 cax=cax).set_label(label='DEM count',size=12)
    if points:
        p, = ax.plot(points[0],points[1],marker='o',color='b',linestyle='none')
    
    ax = axes[1]
    cmap = plt.cm.get_cmap(std_cmap)
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    
    plt.colorbar(ax.imshow(count_nmad_ma_stack[1],
                           vmin=std_vmin,
                           vmax=std_vmax,
                           interpolation='none',
                           alpha=alpha,
                          cmap=cmap),
                 cax=cax).set_label(label='STD [m]',size=12)
    
    if points:
        p, = ax.plot(points[0],points[1],marker='o',color='b',linestyle='none')

    if ticks_off:
        for ax in axes:
            ax.set_xticks(())
            ax.set_yticks(())
        
###########  Models

def remove_nan_from_training_data(X_train, y_train_masked_array):
    array = y_train_masked_array.data
    mask = ~y_train_masked_array.mask                       
    X_train = X_train[mask]
    y_train = y_train_masked_array[mask]
    return X_train, y_train


def create_prediction_timeseries(start_date = '2000-01-01',
                                 end_date = '2023-01-01',
                                 dt ='M'):
    #M  = monthly frequency
    #3M = every 3 months
    #6M = every 6 months
    d = pd.date_range(start_date,end_date,freq=dt)
    X = d.to_series().apply([convert_date_time_to_decimal_date]).values.squeeze()
    return X

def linreg_predict(X_train,
                   y_train,
                   X,
                   method='TheilSen'):
    
    if method=='Linear':
        m = linear_model.LinearRegression()
        m.fit(X_train.squeeze()[:,np.newaxis], y_train.squeeze())
        slope = m.coef_
        intercept = m.intercept_
        prediction = m.predict(X.squeeze()[:,np.newaxis])
        
    if method=='TheilSen':
        m = linear_model.TheilSenRegressor()
        m.fit(X_train.squeeze()[:,np.newaxis], y_train.squeeze())
        slope = m.coef_
        intercept = m.intercept_
        prediction = m.predict(X.squeeze()[:,np.newaxis])
        
    if method=='RANSAC':
        m = linear_model.RANSACRegressor()
        m.fit(X_train.squeeze()[:,np.newaxis], y_train.squeeze())
        slope = m.estimator_.coef_
        intercept = m.estimator_.intercept_
        prediction = m.predict(X.squeeze()[:,np.newaxis])
    
    return prediction, slope[0], intercept

def linreg_run(args):
    X_train, y_train_masked_array, X,  method = args
    
    X_train, y_train = remove_nan_from_training_data(X_train, y_train_masked_array)
    prediction, slope, intercept = linreg_predict(X_train,
                                                  y_train,
                                                  X,
                                                  method='Linear')
    
    return prediction

def linreg_reshape_parallel_results(results, ma_stack, valid_mask_2D):
    results_stack = []
    for i in range(results.shape[1]):
        m = np.ma.masked_all_like(ma_stack[0])
        m[valid_mask_2D] = results[:,i]
        results_stack.append(m)
    results_stack = np.ma.stack(results_stack)
    return results_stack

def linreg_run_parallel(X_train, ma_stack, X, method='Linear'):
    pool = mp.Pool(processes=psutil.cpu_count(logical=True))
    args = [(X_train, ma_stack[:,i], X, method) for i in range(ma_stack.shape[1])]
    results = pool.map(linreg_run, args)
    return np.array(results)

def GPR_kernel():
    v = 10.0
    long_term_trend_kernel = v**2 * RBF(length_scale=v)

    seasonal_kernel = (
        2.0 ** 2
        * RBF(length_scale=100.0)
        * ExpSineSquared(length_scale=1.0, periodicity=1.0, periodicity_bounds="fixed")
    )

    irregularities_kernel = 0.5 ** 2 * RationalQuadratic(length_scale=1.0, alpha=1.0)

    noise_kernel = 0.1 ** 2 * RBF(length_scale=0.1) + WhiteKernel(
        noise_level=0.1 ** 2, noise_level_bounds=(1e-5, 1e5)
    )

    kernel = (
        long_term_trend_kernel + seasonal_kernel + irregularities_kernel + noise_kernel
    )
    return kernel

def GPR_model(X_train, y_train, alpha=1e-10):
    X_train = X_train.squeeze()[:,np.newaxis]
    y_train = y_train.squeeze()
    kernel = GPR_kernel()
    
    gaussian_process_model = GaussianProcessRegressor(kernel=kernel, 
                                                      normalize_y=True,
                                                      alpha=alpha,
                                                      n_restarts_optimizer=9)
    
    gaussian_process_model = gaussian_process_model.fit(X_train, y_train)
    
    return gaussian_process_model

def GPR_predict(gaussian_process_model, X):
    X = X.squeeze()[:,np.newaxis]
    mean_prediction, std_prediction = gaussian_process_model.predict(X, return_std=True)
    
    return mean_prediction, std_prediction

def GPR_run(args):
    X_train, y_train_masked_array, X,  method = args
    X_train, y_train = remove_nan_from_training_data(X_train, y_train_masked_array)
    gaussian_process_model = GPR_model(X_train, y_train, alpha=1e-10)
    prediction, std_prediction = GPR_predict(gaussian_process_model, X)

    return prediction

def GPR_run_parallel(X_train, ma_stack, X, method='Linear'):
    pool = mp.Pool(processes=psutil.cpu_count(logical=True))
    args = [(X_train, ma_stack[:,i], X, method) for i in range(ma_stack.shape[1])]
    results = pool.map(GPR_run, args)
    return np.array(results)

def GPR_reshape_parallel_results(results, ma_stack, valid_mask_2D):
    results_stack = []
    for i in range(results.shape[1]):
        m = np.ma.masked_all_like(ma_stack[0])
        m[valid_mask_2D] = results[:,i]
        results_stack.append(m)
    results_stack = np.ma.stack(results_stack)
    return results_stack

###########  Miscellaneous
def check_if_number_even(n):
    '''
    checks if int n is an even number
    '''
    if (n % 2) == 0:
        return True
    else:
        return False

def make_number_even(n):
    '''
    adds 1 to int n if odd number
    '''
    if check_if_number_even(n):
        return n
    else:
        return n +1

def get_row_column(n):
    '''
    returns largest factor pair for int n
    makes rows the larger number
    '''
    max_pair = max([(i, n / i) for i in range(1, int(n**0.5)+1) if n % i == 0])
    rows = int(max(max_pair))
    columns = int(min(max_pair))
    
    # in case n is odd
    # check if you get a smaller pair by adding 1 to make number even
    if not check_if_number_even(n):
        n = make_number_even(n)
        max_pair = max([(i, n / i) for i in range(1, int(n**0.5)+1) if n % i == 0])
        alt_rows = int(max(max_pair))
        alt_columns = int(min(max_pair))
        
        if (rows,columns) > (alt_rows, alt_columns):
            return (alt_rows, alt_columns)
        else:
            return (rows,columns)
    return (rows,columns)