import numpy as np
cimport numpy as np
cimport cython
import datetime
import sys
import time
import scipy.signal
import scipy.ndimage
import sunpy.io.fits
import queue
import types
from collections import defaultdict
from pathlib import Path
from astropy.io.fits import CompImageHDU
from model import Record
import skimage.measure
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.cluster import KMeans


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        sys.stdout.write(
            '{} : {} ms\n'.format(
                method.__name__, (te - ts) * 1000
            )
        )
        return result
    return timed


@timeit
def do_median_blur(image, kernel_size=3):
    if kernel_size == 0:
        return image.copy()
    return scipy.signal.medfilt2d(image, kernel_size=kernel_size)


@timeit
def do_segmentation_and_label(image, level=256, step_size=1, kernel_size=3):
    median_filtered_image = do_median_blur(
        image,
        kernel_size=kernel_size
    )

    max_image_value = np.nanmax(image)
    if np.isinf(max_image_value):
        raise Exception('Infinite value in image')

    min_step = max_image_value / level
    scaled_step_size = step_size * min_step

    feature_map = np.zeros_like(image)
    no_of_features = 0

    counter = 0

    while counter <= max_image_value:
        min_value = counter
        max_value = counter + scaled_step_size
        temp_map = np.zeros_like(image)
        temp_map[median_filtered_image >= min_value] = 1.0
        temp_map[median_filtered_image > max_value] = 0.0
        _fmap, _no_of_f = scipy.ndimage.label(temp_map)

        _fmap += no_of_features

        _fmap[_fmap == no_of_features] = 0.0

        feature_map = np.add(
            feature_map,
            _fmap
        )

        no_of_features += _no_of_f

        counter += scaled_step_size

    return feature_map, no_of_features


cdef tuple get_intensity_to_o_of_pixel_curve_in_c(np.ndarray data):
    cdef np.ndarray one_d_intensities_arr = data.reshape(
        (data.shape[0] * data.shape[1], )
    )

    cdef list no_of_pixels = list()

    cdef list one_d_intensities = list(set(one_d_intensities_arr))

    one_d_intensities.sort()

    one_d_intensities.remove(0)

    cdef int total = len(one_d_intensities)

    cdef int i = 0

    cdef int percentage = 0

    cdef int old_percentage = -1

    for intensity in one_d_intensities:
        no_of_pixels.append(
            len(np.where(data > intensity)[0])
        )

        percentage = int(
            float(i * 100) / total
        )

        if percentage - old_percentage == 1:
            print ('Percentage Done: {}'.format(percentage))

        old_percentage = percentage

        i += 1

    return (one_d_intensities, no_of_pixels)


@timeit
def get_intensity_to_o_of_pixel_curve(data):
    return get_intensity_to_o_of_pixel_curve_in_c(data)


cdef list get_neighbor(tuple pixel, int shape_1, int shape_2, int type=4):

    pixel_list = list()

    four_connected = [
        (pixel[0] - 1, pixel[1]),
        (pixel[0], pixel[1] - 1),
        (pixel[0] + 1, pixel[1]),
        (pixel[0], pixel[1] + 1)
    ]

    eight_connected = [
        (pixel[0] - 1, pixel[1] - 1),
        (pixel[0] - 1, pixel[1] + 1),
        (pixel[0] + 1, pixel[1] - 1),
        (pixel[0] + 1, pixel[1] + 1)
    ]

    for _pixel in four_connected:
        if 0 <= _pixel[0] < shape_1 and 0 <= _pixel[1] < shape_2:
            pixel_list.append(_pixel)

    if type == 8:
        for _pixel in eight_connected:
            if 0 <= _pixel[0] < shape_1 and 0 <= _pixel[1] < 2:
                pixel_list.append(_pixel)

    return pixel_list


cdef np.ndarray relaxed_pore_image_in_c(np.ndarray image, float threshold):

    points_with_min_intensity = np.where(
        image[200:800, 200:1000] == np.nanmin(image[200:800, 200:1000])
    )

    intensity_set = set()

    checking_dict = defaultdict(dict)

    intensity_set.add(np.nanmin(image))

    a = points_with_min_intensity

    seed_pixel = a[0][0] + 200, a[1][0] + 200

    segment = np.zeros_like(image, dtype=np.int64)

    visited = np.zeros_like(image)

    visiting_queue = queue.Queue()

    visiting_queue.put(seed_pixel)

    checking_dict[seed_pixel[0]][seed_pixel[1]] = 1.0

    while not visiting_queue.empty():

        element = visiting_queue.get()

        visited[element] = 1.0

        if image[element] < threshold:
            segment[element] = 1.0

            neighbors = get_neighbor(
                element,
                image.shape[0],
                image.shape[1]
            )

            for neighbor in neighbors:
                if visited[neighbor] == 0.0:
                    if neighbor[1] not in checking_dict[neighbor[0]]:
                        visiting_queue.put(neighbor)
                        checking_dict[neighbor[0]][neighbor[1]] = 1.0

    return segment


@timeit
def relaxed_pore_image(image, threshold):
    return relaxed_pore_image_in_c(image, threshold)


cdef get_datetime_in_c(file):
    name_string = file.name
    name_string_split = name_string.split('_')

    return datetime.datetime(
        year=int(name_string_split[2][0:4]),
        month=int(name_string_split[2][4:6]),
        day=int(name_string_split[2][6:8]),
        hour=int(name_string_split[3][0:2]),
        minute=int(name_string_split[3][2:4]),
        second=int(name_string_split[3][4:6])
    )


@timeit
def get_datetime_from_name(file):
    return get_datetime_in_c(file)


def actual_size_mean_min_intensity(data, mask):
    mask = mask.astype(np.bool)
    image = data * mask
    total_pixels = np.sum(mask)
    total_intensity = np.sum(image)
    mean_intensity = float(total_intensity) / total_pixels
    image[image == 0.0] = np.nan
    return total_pixels, mean_intensity, np.nanmin(image)


def size_mean_min_intensity(data_path, segment_path):
    data, _ = sunpy.io.fits.read(data_path)[0]
    mask, _ = sunpy.io.fits.read(segment_path)[1]
    return actual_size_mean_min_intensity(data, mask)


def calculate_rest_of_stuff(base_path, segment_base_path):
    everything = base_path.glob('**/*')
    files = [x for x in everything if x.is_file() and x.name.endswith('.fts')]

    for file in files:
        name = file.name + '.segment.fits'
        segment_path = segment_base_path / name
        total_pixels, mean_intensity, min_value = size_mean_min_intensity(
            file, segment_path
        )
        record = Record.find_by_date(get_datetime_from_name(file))
        record.size = total_pixels
        record.mean_intensity = mean_intensity
        record.min_intensity = min_value
        record.save()


def wrapper_to_initialise(rr, cc, values):

    mapper_dict = defaultdict(dict)

    for r, c, v in zip(rr, cc, values):
        mapper_dict[r][c] = v

    def initialize_my_array(x, y):
        if y in mapper_dict[x]:
            return mapper_dict[x][y]
        return 0.0

    return initialize_my_array


def subpartition(image, mask):
    rr, cc = np.where(mask.astype(np.bool))

    nrr = rr.reshape((len(rr), 1))
    ncc = cc.reshape((len(cc), 1))

    X = np.concatenate((nrr, ncc), axis=1)

    kmeans = KMeans(n_clusters=3)

    kresult = kmeans.fit(
        X,
        sample_weight=image[rr, cc].reshape(
            (len(rr), )
        )
    )

    labels = kresult.labels_ + 1

    function_to_initialise = wrapper_to_initialise(
        rr, cc, labels.reshape((labels.shape[0], ))
    )

    victor = np.vectorize(function_to_initialise)

    new_mask = np.fromfunction(
        victor,
        shape=image.shape,
        dtype=int
    )

    return new_mask


@timeit
def do_all(base_path, write_path):
    everything = base_path.glob('**/*')
    files = [x for x in everything if x.is_file() and x.name.endswith('.fts')]

    for file in files:
        image, _ = sunpy.io.fits.read(file)[0]
        hist_data = np.histogram(image[200:800, 200:1000])
        segment = relaxed_pore_image(image, hist_data[1][3])
        kl = segment * image * 0.2 + image
        write_file_path_image_identify = write_path / file.name
        sunpy.io.fits.write(
            write_file_path_image_identify, kl, dict(), hdu_type=CompImageHDU
        )
        segment_path = file.name + '.segment.fits'
        write_file_path_image_segment = write_path / segment_path
        sunpy.io.fits.write(
            write_file_path_image_segment, segment.astype(np.float64),
            dict(), hdu_type=CompImageHDU
        )
        a, b, c = actual_size_mean_min_intensity(
            image, segment
        )
        total_pixels, mean_intensity, min_value = a, b, c

        record = Record(
            date_time=get_datetime_from_name(file)
        )

        section_mask = subpartition(image, segment)

        subsegment_path = file.name + '.subsegment.fits'
        write_file_path_image_subsegment = write_path / subsegment_path
        sunpy.io.fits.write(
            write_file_path_image_subsegment, section_mask,
            dict(), hdu_type=CompImageHDU
        )

        section_region_props = skimage.measure.regionprops(
            label_image=section_mask.astype(int),
            intensity_image=image,
            cache=True
        )

        name_dict = {
            1: 'one',
            2: 'two',
            3: 'three'
        }

        i = 1
        for subregion in section_region_props:
            _this_segment = (section_mask == i) * i
            a, b, c = actual_size_mean_min_intensity(
                image, _this_segment
            )

            _size, _mean, _min = a, b, c

            _var = 'section_' + name_dict[i]
            setattr(record, _var + '_size', _size)
            setattr(record, _var + '_mean_intensity', _mean)
            setattr(record, _var + '_min_intensity', _min)
            setattr(record, _var + '_eccentricity', subregion.eccentricity)

            i += 1

        region_props = skimage.measure.regionprops(
            label_image=segment,
            intensity_image=image,
            cache=True
        )

        for region in region_props:

            record.threshold = hist_data[1][3]
            record.eccentricity = region.eccentricity
            record.size = total_pixels
            record.mean_intensity = mean_intensity
            record.min_intensity = min_value
        record.save()


if __name__ == '__main__':
    filename = '/Users/harshmathur/Documents/' + \
        'CourseworkRepo/PoreAnalyis/data/' + \
        'filtrd_hifi_20170928_085120_sd_speckle.fts'

    image, header = sunpy.io.fits.read(filename)[0]

    image = image.byteswap().newbyteorder()

    feature_map, no_of_features = do_segmentation_and_label(
        image, step_size=10
    )

    sunpy.io.fits.write(
        'feature_map.fits',
        feature_map,
        {'no_of_features': no_of_features}
    )
