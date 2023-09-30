import rasterio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.widgets import Slider
from os import listdir
from alive_progress import alive_bar
from rasterio.windows import Window
from pyproj import Proj
from math import floor, ceil
import h5py
import xml.etree.ElementTree as ET

def getTiff(path, file, dtype):

    '''
    getTiff(path, file, dtype)
    
    path: directory to your tiff file
    filename: file name of your tiff file including type of file (.tif etc)
    type of filename have to be list
    dtype: 'grd' for GRD files, 'slc' for slc files
    '''
    if dtype == 'grd':
        with rasterio.open(path + file) as src:
            img = src.read()
            img_meta = src.meta
            bound = src.bounds
            
    if dtype == 'slc':
        with rasterio.open(path + file) as src:
            img0 = src.read()
            img = img0[:,:,0] + img0[:,:,1] * 1j
            img_meta = src.meta
            bound = src.bounds
            
    return {'Band': np.array(img)[0],
            'Product Name': file,
            'Meta Data': img_meta, 
            'coords': (bound[0], bound[1], bound[2], bound[3])}

def getK5h5 (path, filename, dtype):
    
    '''
    return complex image SLC band and Calibration Factors of kompsat-5
    getK5 (path, file)
    
    path: directory to your h5 file of KOMPSAT-5
    file: file name of your h5 file of KOMPSAT-5
    '''
    if dtype == 'slc':
        file = h5py.File(path+filename, 'r')
        img = file['/S01/SBI']
        
        img  = {'Band': img[:,:,0] + img[:,:,1] * 1j,
                'Platform': file.attrs['Satellite ID'],
                'Product Name': filename,
                'Polarization': file['/S01'].attrs['Polarisation'],
                'Orbit Direction': file.attrs['Orbit Direction'],
                'Frequency': file.attrs['Radar Frequency'],
                'Wavelength': file.attrs['Radar Wavelength'],
                'Azimuth Resolution': file.attrs['Azimuth Geometric Resolution'],
                'Ground Range Resolution': file.attrs['Ground Range Geometric Resolution'],
                'Scene Centre Geodetic Coordinates': file.attrs['Scene Centre Geodetic Coordinates'],
                'Coords': (file['/S01/SBI'].attrs['Top Left Geodetic Coordinates'],
                           file['/S01/SBI'].attrs['Top Right Geodetic Coordinates'],
                           file['/S01/SBI'].attrs['Bottom Left Geodetic Coordinates'],
                           file['/S01/SBI'].attrs['Bottom Right Geodetic Coordinates']),
                'Ellipsoid': file.attrs['Ellipsoid Designator'],
                'Calibration Factors': {'Calibration Constant': file['/S01'].attrs['Calibration Constant'],
                                                     'Rescaling Factor': file.attrs['Rescaling Factor'],
                                                     'Reference Incidence Angle': file.attrs['Reference Incidence Angle']}
                
                }
        file.close()
        del file
        return img
    

    if dtype == 'grd':
        file = h5py.File(path+filename, 'r')
        img = file['/S01/SBI']
        
        img = {'Band': img,
                'Platform': file.attrs['Satellite ID'],
                'Product Name': filename,
                'Polarization': file['/S01'].attrs['Polarisation'],
                'Orbit Direction': file.attrs['Orbit Direction'],
                'Azimuth Resolution': file.attrs['Azimuth Geometric Resolution'],
                'Ground Range Resolution': file.attrs['Ground Range Geometric Resolution'],
                'Scene Centre Geodetic Coordinates': file.attrs['Scene Centre Geodetic Coordinates'],
                'Coords': (file['/S01/SBI'].attrs['Top Left Geodetic Coordinates'],
                           file['/S01/SBI'].attrs['Top Right Geodetic Coordinates'],
                           file['/S01/SBI'].attrs['Bottom Left Geodetic Coordinates'],
                           file['/S01/SBI'].attrs['Bottom Right Geodetic Coordinates']),
                'Ellipsoid': file.attrs['Ellipsoid Designator'],
                'Calibration Factors': {'Calibration Constant': file['/S01'].attrs['Calibration Constant'],
                                                     'Rescaling Factor': file.attrs['Rescaling Factor'],
                                                     'Reference Incidence Angle': file.attrs['Reference Incidence Angle']}
                
                }
        del file
        return img

def getProduct(path, filename, format, dtype):

    if format == 'K5':
        return getK5h5(path, filename, dtype)
    
    if format == 'K5tif':
        return print('not available yet')
    
    if format == 'S1tiff':
        if dtype == 'grd':
            for i in range (0, len(listdir(path[:-len('measurement/')] + 'annotation/'))):
                if listdir(path[:-len('measurement/')] + 'annotation/')[i][-7:-4] == filename[-8:-5]:
                    xml = listdir(path[:-len('measurement/')] + 'annotation/')[i]

            tree = ET.parse(path[:-len('measurement/')] + 'annotation/' + xml)
            root = tree.getroot()
            data = {'Band': [],
                    'Platform': root.find('adsHeader').find('missionId').text,
                    'Product Name': filename,
                    'Polarization': root.find('adsHeader').find('polarisation').text,
                    'Orbit Direction': root.find('generalAnnotation').find('productInformation').find('pass').text,
                    'Azimuth Resolution': root.find('imageAnnotation').find('imageInformation').find('azimuthPixelSpacing').text,
                    'Ground Range Resolution': root.find('imageAnnotation').find('imageInformation').find('rangePixelSpacing').text,
                    'Scene Centre Geodetic Coordinates': [],
                    'Coords': [(root.find('geolocationGrid')[0][0].find('longitude').text,root.find('geolocationGrid')[0][0].find('latitude').text),
                               (root.find('geolocationGrid')[0][20].find('longitude').text,root.find('geolocationGrid')[0][20].find('latitude').text),
                               (root.find('geolocationGrid')[0][-21].find('longitude').text,root.find('geolocationGrid')[0][-21].find('latitude').text),
                               (root.find('geolocationGrid')[0][-1].find('longitude').text,root.find('geolocationGrid')[0][-1].find('latitude').text)],
                    'Ellipsoid': [],
                    'Calibration Factors': []
                    }
            

            with rasterio.open(path + filename) as src:
                img = src.read()
                img_meta = src.meta
                bound = src.bounds
            data['Band'] = img[0]
            data['Ellipsoid'] = src.crs
            src.close()

            return data

        if dtype == 'slc':
            return print('not available yet')
        



def longlat2window(lon, lat, dataset):
    """
    Args:
        lon (tuple): Tuple of min and max lon
        lat (tuple): Tuple of min and max lat`
        dataset: Rasterio dataset

    Returns:
        rasterio.windows.Window``
    """
    p = Proj(dataset.crs)
    t = dataset.transform
    xmin, ymin = p(lon[0], lat[0])
    xmax, ymax = p(lon[1], lat[1])
    col_min, row_min = ~t * (xmin, ymin)
    col_max, row_max = ~t * (xmax, ymax)
    return Window.from_slices(rows=(floor(row_max), ceil(row_min)),
                              cols=(floor(col_min), ceil(col_max)))

def getSubset (path, file, lon, lat, platform):
    
    '''
    return subset image band and meta data of geotiff file as dictionary
    getSubset (path, file, lon, lat, dtype)
    
    path : directory to your Geotiff file
    file : your file name including '.tif'
    file must be list
    dtype: 'grd' for GRD files, 'slc' for slc files
    lon, lat must be tuple
    '''

    with rasterio.open(path + file) as src:
        window = longlat2window(lon, lat, src)
        img = src.read(window = window)
        img_meta = src.meta
        p = Proj(src.crs)
        t = src.transform
        xmin, ymin = p(lon[0], lat[0])
        xmax, ymax = p(lon[1], lat[1])
        src.close()
    
        
            
    return {'Band': np.array(img)[0], 
            'Platform': platform,
            'Product Name': file,
            'Meta Data': img_meta, 
            'Coords': (xmin, ymin, xmax, ymax)}


def plotall (img_list, Clim):

    '''
    show all of input images

    img_list has to be only one band image stack
    This function doesn't treat multi band image stack
    Clim: color axis, have to be tuple: (min, max)
    '''
    
    from matplotlib.colors import LinearSegmentedColormap
    colors = ['#000000', '#149494', '#1E9E9E','#2A8A8A', '#32B2B2',
              '#37B7B7', '#3CBCBC', '#41C1C1', '#FFFFFF']
    cmap = LinearSegmentedColormap.from_list('my_cmap',colors,gamma=2)
    cmap
    
    for i in range (0, len(img_list)):
        plt.figure(figsize = (15,15))
        plt.subplot(int(len(img_list)/4)+1, 4, i+1)
        fig = plt.imshow(img_list[i], cmap = cmap)
        fig.set_clim(Clim)
        plt.axis('off')

    return plt.show()


def imgshow (img, clim, title):

    '''
    show input image

    img: only one image is available (2D array)
    clim: set color axis as tuple
    title: set image title
    '''
    from matplotlib.colors import LinearSegmentedColormap
    #colors = ['#000000', '#149494', '#1E9E9E','#2A8A8A', '#32B2B2',
    #          '#37B7B7', '#3CBCBC', '#41C1C1', '#FFFFFF']
    colors = ['#000000','#162929', '#1F4F4E', '#247777', '#25A2A2', '#63D4D3', '#1ED0D0',
              '#00FFFF', '#67FFFF', '#91FFFF', '#B2FFFF', '#CEFFFF', '#E7FFFF', '#FFFFFF']
    
    cmap = LinearSegmentedColormap.from_list('my_cmap',colors,gamma=2)
    cmap

    #plt.figure(figsize = (15,15))
    def onclick(event):
        if event.button == 1:
            print(f'pixel coords: x={int(event.xdata)}, y={int(event.ydata)}')

    fig, ax = plt.subplots()
    a = plt.imshow(img, cmap = cmap)
    a.set_clim(clim)
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.title(title, size = 20)

    return plt.show()

def imgshow_sq(stack, clim, title):

    num_images = len(stack)
    current_index = 0


    imlist = [item['Band'] for item in stack]

    colors = ['#000000','#162929', '#1F4F4E', '#247777', '#25A2A2', '#63D4D3', '#1ED0D0',
              '#00FFFF', '#67FFFF', '#91FFFF', '#B2FFFF', '#CEFFFF', '#E7FFFF', '#FFFFFF']
    cmap = LinearSegmentedColormap.from_list('my_cmap', colors, gamma=2)


    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.15)

    image = ax.imshow(imlist[current_index], cmap=cmap)
    image.set_clim(clim)
    plt.title(title, size=20)

    axcolor = 'lightgoldenrodyellow'
    ax_slider = plt.axes([0.2, 0.01, 0.65, 0.03], facecolor=axcolor)
    plt.title(stack[0]['Product Name'][:15])
    slider = Slider(ax_slider, 'Image', 0, num_images - 1, valinit=current_index, valstep=1)

    

    def update(val):
        index = int(slider.val)
        image.set_data(imlist[index])
        fig.canvas.draw_idle()
        plt.title(stack[index]['Product Name'][:15])

    slider.on_changed(update)

    plt.show()

def imgshow_ts (stack, clim, title):
 
    imlist = []
    for i in range (0, len(stack)):
         np.array(imlist.append(stack[i]['Band']))
 
    from matplotlib.colors import LinearSegmentedColormap
    colors = ['#000000','#162929', '#1F4F4E', '#247777', '#25A2A2', '#63D4D3', '#1ED0D0',
              '#00FFFF', '#67FFFF', '#91FFFF', '#B2FFFF', '#CEFFFF', '#E7FFFF', '#FFFFFF']
    cmap = LinearSegmentedColormap.from_list('my_cmap',colors,gamma=2)
    cmap

    #plt.figure(figsize = (15,15))
    def onclick(event):
        if event.button == 1:
            print(f'pixel coords: x={int(event.xdata)}, y={int(event.ydata)}')
            signal = getSignal(imlist, int(event.ydata), int(event.xdata))
            x = generate_x(imlist)

            plt.figure()
            plt.plot(x, signal, '.-', label = 'ts_signal')
            plt.xlabel('Data Aqcuisition Sequence')
            plt.ylabel('Pixel Value')
            plt.legend()
            plt.title('Time Series Pixel Value')
            plt.show()

    fig, ax = plt.subplots()
    a = plt.imshow(temp_avg(imlist), cmap = cmap)
    a.set_clim(clim)
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.title(title, size = 20)
 
    return plt.show()

def show_result(result, background, type):

    if type == 'avg':

        img = []
        for i in range (0, len(background)):
            img.append(background[i]['Band'])

        avg = temp_avg(img)
        fig = plt.imshow(avg, cmap = 'gray')
        fig.set_clim(0,2)
        fig = plt.imshow(result['Band'], cmap = 'bwr', alpha = 0.5)
        fig.set_clim(-1, 1)

    if type == 'match':
        fig = plt.imshow(background['Band'], cmap = 'gray')
        fig.set_clim(0,2)
        fig = plt.imshow(result['Band'], cmap = 'bwr', alpha = 0.5)
        fig.set_clim(-1, 1)

    return plt.show()

def getHist (img, bins):

    '''
    return histogram of input image

    img: only one image is available
    bins: set histogram axis limit
    Ex) bins = np.arange(0.01, 1, 0.01)
    '''
    
    y, x = np.histogram(img, bins = bins)
    y = y / np.sum(y)
    plt.plot(x[0:-1], y, '.', label = 'Data')
    plt.legend()
    plt.title('Histogram', size = 20)
    return plt.show()


def getInt (data):
    
    '''
    return intensity image

    data: Single Look Complex product
    '''
    
    result = data.copy()
    intensity = abs(result['Band'])**2
    result['Band'] = intensity
    return result



def getAmp (data):

    '''
    return amplitude image
    getAmp (data)
    
    data: Single Look Complex product
    slc[0] : real number component of slc image
    slc[1] : imaginary number component of slc image
    '''

    result = data.copy
    amplitude = abs(result['Band'])
    result['Band'] = amplitude
    return result
    

'''
def getGRD (slc):
'''


def getdB (data):

    '''
    return decibell scale SAR image
    getdB (data)
    
    data: slc product is not available
    '''
    
    result = data.copy()
    img = result['Band']
    img[img == 0] = 0.00001
    dB = 10 * np.log10(img)
    result['Band'] = dB
    return result

def getRGB (red, green, blue):
    
    '''
    return RGB scale image
    
    getRGB (red, green blue)
    
    red: input image represented as red
    green: input image represented as green
    blue: input image represented as blue
    '''
    
    RGB = np.dstack((red, green))
    RGB = np.dstack((RGB, blue))
    
    return RGB

def calibration (data, platform):
    
    '''
    return sigma0 image of SLC image
    calibrtion (data, platform)
    
    data: complex SLC data with calibration factors
    platform: 'K5' for KOMPSAT-5, 'S1' for Sentinel-1
    '''
        
    if platform == 'K5':
        result = data.copy()
        cf = result['Calibration Factors']
        s = result['Band']
        cal = abs(s * cf['Rescaling Factor'])**2 * cf['Calibration Constant'] * np.sin(cf['Reference Incidence Angle'] * 0.017)
        result['Band'] = cal
    
    if platform == 'S1':
        result = data.copy()
        cal = result['Band'] / (result['Azimuth Resolution'] * result['Ground Range Resolution'])
        result['Band'] = cal

    return result

'''
from imgreg.models.logpolar import LogPolarSolver

def imgStack (im1, im2):
    
    ref_img = np.array(im1)
    mod_img = np.array(im2)
    lps = LogPolarSolver(ref_img, mod_img)
    
    ref = lps.REF_IMG.value
    reg = lps.RECOVERED_ROT_SCALE_TR_IMG.value
    
    return [ref, reg]

def getStackParm (im1, im2):
    
    ref_img = np.array(im1)
    mod_img = np.array(im2)
    lps = LogPolarSolver(ref_img, mod_img)
    
    ROTAT = lps.RECOVERED_ROTATION.value
    TRANS = lps.RECOVERED_TRANSLATION.value
    SCALE = lps.RECOVERED_SCALE.value
    
    return ROTAT, TRANS, SCALE

def slcStack (slc1, slc2, ROTAT, TRANS, SCALE):
    
    ref_img = np.array(slc1)
    mod_img = np.array(slc2)
    lps = LogPolarSolver(ref_img, mod_img)
    
    lps.RECOVERED_ROTATION.value = ROTAT
    lps.RECOVERED_TRANSLATION.value = TRANS
    lps.RECOVERED_SCALE.value = SCALE

    
    ref = lps.REF_IMG.value
    reg = lps.RECOVERED_ROT_SCALE_TR_IMG.value
    
    return [ref, reg]
'''
#%% target detection
'''
def thresholding (img):
'''

def cfar(data, alpha, win_size):
    '''
    return CFAR algorithm applied product

    data: input product
    alph: False alarm rate
    win_size: must be integer odd number
    '''

    im = data.copy()
    a, b = im['Band'].shape
    target_pixel = win_size // 2 + 1
    TD = np.zeros(np.shape(im['Band']))

    with alive_bar(a - win_size, force_tty = True) as bar:
        for i in range (0, a - win_size):
            for j in range (0, b - win_size):
                win = im['Band'][i:i+win_size, j:j+win_size]
                clutter = []
                clutter.append(win[:-1, 0].flatten())
                clutter.append(win[-1, :-1].flatten())
                clutter.append(win[1:, -1].flatten())
                clutter.append(win[0, 1:].flatten())

                bins = np.arange(0, 20, 0.01)
                y, x = np.histogram(clutter, bins = bins, range = (0, 10))
                y = y / np.sum(y)
                fft = np.fft.fft(y)
                fft[30:-30] = 0
                fitted = np.abs(np.fft.ifft(fft))
                p_value = ppf(fitted, x, 1-alpha)

                if win[target_pixel, target_pixel] > p_value:
                    TD[i+target_pixel, j+target_pixel] = 1
            bar()
    im['Band'] = TD

    return im

def ca_cfar(product, GUARD_CELLS, BG_CELLS, ALPHA):
    '''
    Copyright:
    https://qiita.com/harmegiddo/items/8a7e1b4b3a899a9e1f0c
    '''
    CFAR_UNITS = 1 + (GUARD_CELLS * 2) + (BG_CELLS * 2)
    

    inputImg = product.copy()
    ship = np.zeros((np.shape(inputImg['Band'])), np.uint8)


    with alive_bar(inputImg['Band'].shape[0] - CFAR_UNITS, force_tty = True) as bar:
        for i in range(inputImg['Band'].shape[0] - CFAR_UNITS):
            center_cell_x = i + BG_CELLS + GUARD_CELLS
            for j in range(inputImg['Band'].shape[1] - CFAR_UNITS):
                center_cell_y = j  + BG_CELLS + GUARD_CELLS
                average = 0
                for k in range(CFAR_UNITS):
                    for l in range(CFAR_UNITS):
                        if (k >= BG_CELLS) and (k < (CFAR_UNITS - BG_CELLS)) and (l >= BG_CELLS) and (l < (CFAR_UNITS - BG_CELLS)):
                            continue
                        average += inputImg['Band'][i + k, j + l]
                average /= (CFAR_UNITS * CFAR_UNITS) - ( ((GUARD_CELLS * 2) + 1) * ((GUARD_CELLS * 2) + 1) )

                if inputImg['Band'][center_cell_x, center_cell_y] > (average * ALPHA):
                    ship[center_cell_x, center_cell_y] = 1
            bar()

    inputImg['Band'] = ship
    return inputImg

#%% change detection

def ppf(pdf, x, p):

    i = 1
    
    while (sum(pdf[0:i]) < p):
        i = i + 1
    
    
    return x[i-1]

def ratio_cd (img_stack, img_sub, alpha, show):
    
    '''
    returns change map between img_stack time interval, and shows PDF histogram of ratio image of img_sub
    
    img_stack : 2 images are required, image have to include sub image
    img_sub : 2 images are required
    alpha : confidance(threshold), 0to 1; 0.05 means 99% confidence
    show : of show is 'True', it shows PDF histogram & the result
    '''
    
    img_stack[0][img_stack[0] == 0] = 0.00001
    img_stack[1][img_stack[1] == 0] = 0.00001
    img_sub[0][img_sub[0] == 0] = 0.00001
    img_sub[1][img_sub[1] == 0] = 0.00001
    
    
    ratio_train = np.array(img_sub[0]) / np.array(img_sub[1])
    bins = np.arange(0, 5, 0.01)
    y, x = np.histogram(ratio_train, bins = bins, range = (0, 5))
    y = y / np.sum(y)
    fft = np.fft.fft(y)
    fft[30:-30] = 0
    fitted = np.abs(np.fft.ifft(fft))
    #plt.plot(x[0:-1], y, '.-', label = 'Data')
    #plt.plot(x[0:-1], fitted, '-', color = 'red')
    #plt.legend()
    #plt.title('Fitted PDF', size = 20)
    #if show == 'True':
    #    plt.show()

    pv = ppf(fitted, x, alpha)
    print('%.2f' %(sum(fitted)))

    ratio1 = img_stack[0] / img_stack[1]
    ratio2 = img_stack[1] / img_stack[0]

    c_map = np.zeros(np.shape(ratio1))

    c_map[ratio1 < pv] = 1
    c_map[ratio2 < pv] = -1
    c_map[c_map == 0] = np.nan
    
    from matplotlib.colors import LinearSegmentedColormap
    colors = ['#00AAFF', '#FFFFFF', '#F05690']
    cmap = LinearSegmentedColormap.from_list('my_cmap',colors,gamma=2)
    cmap
    
    plt.figure(figsize = (15,15))
    fig = plt.imshow(img_stack[0], cmap = 'gray')
    fig.set_clim(0, 3)
    fig = plt.imshow(c_map, cmap = cmap)
    fig.set_clim(-1, 1)
    plt.axis('off')
    plt.title('Detected Change', size = 20)
    if show == 'True':
        plt.show()

    return c_map

'''
dual pol
'''
#%% speckle filtering
'''
def MultiLook (img, win):
'''
            
            
from scipy.ndimage.filters import uniform_filter
from scipy.ndimage.measurements import variance

def lee_filter(data, size):
    
    '''
    this algorithm is based on site, linked below
    https://stackoverflow.com/questions/39785970/speckle-lee-filter-in-python

    return filtered image of img using Lee Filter method
    
    data : only one product is available
    size : kernel size(odd number)
    if size = 3, kernel size is 3x3
    '''
    
    result = data.copy()
    img = result['Band']
    img_mean = uniform_filter(img, (size, size))
    img_sqr_mean = uniform_filter(img**2, (size, size))
    img_variance = img_sqr_mean - img_mean**2

    overall_variance = variance(img)

    img_weights = img_variance / (img_variance + overall_variance)
    img_output = img_mean + img_weights * (img - img_mean)
    result['Band'] = img_output
    return result

def temp_avg (img):
    
    '''
    return temporal average image
    temp_avg (img)
    
    img: image stack
    '''
    mean_img = sum(img) / len(img)
    return mean_img


from PyEMD import EMD


def getSignal(im_list, a, b):
    signal = []
    for i in range(len(im_list)):
        signal.append(im_list[i][a, b])
    return signal

def generate_x(vvlist):
    x = []
    for i in range (0, len(vvlist)):
        x.append(i)
    return x

def EMD_filter (imlist, n):
    '''
    return EMD(Empirical Mode Decomposition) Applied multi temporal images
    EMD_filter (imlist, n)
    
    imlist : multi-temporal image stack, registration must be done
    n: number of IMFs(1 to 3) to substract from original signal (which determines filtering effect)
    '''
    
    newimg = imlist
    a, b, c = np.shape(imlist)
    if n == 1:
        with alive_bar(b, force_tty = True) as bar:
            for i in range (0, b):
                for j in range (0, c):
                    
                    signal = getSignal(imlist, i, j)
                    x = generate_x(imlist)
                    emd = EMD()
                    imfs = emd.emd(np.array(signal), np.array(x))
                    denoised_signal = signal - imfs[0]
                    for k in range (0, a):
                        newimg[k][i,j] = denoised_signal[k]
                bar()
                        
    if n == 2:                    
        with alive_bar(b, force_tty = True) as bar:
            for i in range (0, b):
                for j in range (0, c):
                    
                    signal = getSignal(imlist, i, j)
                    x = generate_x(imlist)
                    emd = EMD()
                    imfs = emd.emd(np.array(signal), np.array(x))
                    denoised_signal = signal - imfs[0] - imfs[1]
                    for k in range (0, a):
                        newimg[k][i,j] = denoised_signal[k]
                bar()
                        
    if n == 3:                    
        with alive_bar(b, force_tty = True) as bar:
            for i in range (0, b):
                for j in range (0, c):
                    
                        signal = getSignal(imlist, i, j)
                        x = generate_x(imlist)
                        emd = EMD()
                        imfs = emd.emd(np.array(signal), np.array(x))
                        denoised_signal = signal - imfs[0] - imfs[1] - imfs[2]
                        for k in range (0, a):
                            newimg[k][i,j] = denoised_signal[k]
                bar()
        return newimg


'''
fft
'''
