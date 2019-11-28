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
from model import Record, PoreData, PoreSectionData
import skimage.measure
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.cluster import KMeans
import astropy.units as u
import sunpy.map


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
def write_to_disk(filename, image, header, hdu_type=None):
    sunpy.io.fits.write(filename, image, header, hdu_type=hdu_type)


@timeit
def do_median_blur(image, kernel_size=3):
    if kernel_size == 0:
        return image.copy()
    return scipy.signal.medfilt2d(image, kernel_size=kernel_size)


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


class PoreSegment(object):
    def __init__(self, segment, points_with_min_intensity):
        self._segment = segment
        self._points_with_min_intensity = points_with_min_intensity

    @property
    def segment(self):
        return self._segment

    @property
    def points_with_min_intensity(self):
        return self._points_with_min_intensity


cdef object relaxed_pore_image_in_c(np.ndarray image, float threshold):

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

    return PoreSegment(
        segment=segment,
        points_with_min_intensity=points_with_min_intensity
    )


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
    total_pixels = np.nansum(mask)
    total_intensity = np.nansum(image)
    mean_intensity = float(total_intensity) / total_pixels
    image[image == 0.0] = np.nan
    return total_pixels, mean_intensity, np.nanmin(image)


'''
def wrapper_to_initialise(rr, cc, values):

    mapper_dict = defaultdict(dict)

    for r, c, v in zip(rr, cc, values):
        mapper_dict[r][c] = v

    def initialize_my_array(x, y):
        if y in mapper_dict[x]:
            return mapper_dict[x][y]
        return 0.0

    return initialize_my_array


def subpartition(image, mask, n_clusters=3):
    rr, cc = np.where(mask.astype(np.bool))

    nrr = rr.reshape((len(rr), 1))
    ncc = cc.reshape((len(cc), 1))

    X = np.concatenate((nrr, ncc), axis=1)

    kmeans = KMeans(n_clusters=n_clusters)

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


def save_for_sub_segment(
    image, header, segment, k, threshold, pore_data, file, write_path
):
    section_mask = subpartition(image, segment, n_clusters=4)

    subsegment_path = file.name + '.' + str(threshold) + '.subsegment.fits'
    write_file_path_image_subsegment = write_path / subsegment_path

    write_to_disk(
        write_file_path_image_subsegment, section_mask,
        dict(), hdu_type=CompImageHDU
    )

    section_region_props = skimage.measure.regionprops(
        label_image=section_mask.astype(int),
        intensity_image=image,
        cache=True
    )

    i = 1

    for subregion in section_region_props:
        _this_segment = (section_mask == i) * i

        pore_section_data = PoreSectionData()

        a, b, c = actual_size_mean_min_intensity(
            image, _this_segment
        )

        _size, _mean, _min = a, b, c

        pore_section_data.pore_data_id = pore_data.id

        pore_section_data.k = k

        pore_section_data.threshold = threshold

        pore_section_data.eccentricity = subregion.eccentricity

        pore_section_data.size = _size

        pore_section_data.mean_intensity = _mean

        pore_section_data.min_intensity = _min

        pore_section_data.save()

        i += 1
'''


def save_for_this_segment(image, header, k, record, file, write_path):

    image[np.where(image < 0)] = 0.0

    qs_mean = np.nanmean(image[160:330, 723:923])

    image = image / qs_mean

    mn = np.nanmean(image)

    sd = np.nanstd(image)

    threshold = get_threshold(mn, sd, k)

    pore_segment = relaxed_pore_image(image, threshold)

    segment, point_with_min_intensity = pore_segment.segment, \
        pore_segment.points_with_min_intensity

    kl = segment * image * 0.2 + image

    filename = file.name + '.' + str(threshold) + '.fits'

    write_file_path_image_identify = write_path / filename

    write_to_disk(
        write_file_path_image_identify, kl, dict()
    )

    gregor_map = sunpy.map.Map(image, header)

    coord = gregor_map.pixel_to_world(
        u.Quantity(
            point_with_min_intensity[0][0],
            u.pixel
        ),
        u.Quantity(
            point_with_min_intensity[1][0],
            u.pixel
        )
    )

    x = coord.heliocentric.x.value

    y = coord.heliocentric.y.value

    z = coord.heliocentric.z.value

    theta = np.arctan(z / np.sqrt((x**2) + (y**2)))

    phi = np.arctan(y / x)

    uncorrected_distance_per_pixel = gregor_map.rsun_meters.value * 0.0253 \
        / gregor_map.rsun_obs.value

    # x is normal x axis, i.e. horizontal, not pythons!!!

    corrected_distance_per_pixel_x = uncorrected_distance_per_pixel \
        / np.cos(phi)

    # y is normal y axis, i.e. vertical, not pythons!!!

    corrected_distance_per_pixel_y = uncorrected_distance_per_pixel \
        / np.cos(theta)

    area_per_pixel = corrected_distance_per_pixel_x * \
        corrected_distance_per_pixel_y

    segment_path = file.name + '.' + str(threshold) + '.segment.fits'
    write_file_path_image_segment = write_path / segment_path
    write_to_disk(
        write_file_path_image_segment, segment.astype(np.float64),
        dict(), hdu_type=CompImageHDU
    )

    a, b, c = actual_size_mean_min_intensity(
        image, segment
    )
    total_pixels, mean_intensity, min_value = a, b, c

    pore_data = PoreData()

    region_props = skimage.measure.regionprops(
        label_image=segment,
        intensity_image=image,
        cache=True
    )

    for region in region_props:
        pore_data.record_id = record.id

        pore_data.k = k

        pore_data.threshold = threshold

        pore_data.eccentricity = region.eccentricity

        pore_data.size = total_pixels * area_per_pixel

        pore_data.mean_intensity = mean_intensity

        pore_data.min_intensity = min_value

        uncorrcted_major_length = region.major_axis_length

        x_component_maj = uncorrcted_major_length * np.cos(region.orientation)

        y_component_maj = uncorrcted_major_length * np.sin(region.orientation)

        corr_x_maj = x_component_maj * corrected_distance_per_pixel_y

        corr_y_maj = y_component_maj * corrected_distance_per_pixel_x

        corr_major_length = np.sqrt(corr_x_maj ** 2 + corr_y_maj ** 2)

        uncorrcted_minor_length = region.minor_axis_length

        x_component_minor = uncorrcted_minor_length * np.cos(
            region.orientation)

        y_component_minor = uncorrcted_minor_length * np.sin(
            region.orientation)

        corr_x_minor = x_component_minor * corrected_distance_per_pixel_y

        corr_y_minor = y_component_minor * corrected_distance_per_pixel_x

        corr_minor_length = np.sqrt(corr_x_minor ** 2 + corr_y_minor ** 2)

        pore_data.major_axis_length = corr_major_length

        pore_data.minor_axis_length = corr_minor_length

        pore_data.inertia_tensor_eigvals = str(
            region.inertia_tensor_eigvals
        )

        pore_data.orientation = region.orientation

        _centroid = gregor_map.pixel_to_world(
            u.Quantity(
                region.weighted_centroid[1],
                u.pixel
            ),
            u.Quantity(
                region.weighted_centroid[0],
                u.pixel
            )
        )

        centroid = (
            _centroid.heliocentric.x.value,
            _centroid.heliocentric.y.value,
            _centroid.heliocentric.z.value
        )

        pore_data.centroid = str(centroid)

    pore_data.save()

    # save_for_sub_segment(
    #     image, header, segment, k, threshold, pore_data, file, write_path
    # )


def get_threshold(mn, sd, k):
    return mn + (k * sd)


@timeit
def read_file(file):
    return sunpy.io.fits.read(file)[0]


@timeit
def populate_qs_mesn_and_std(base_path):
    everything = base_path.glob('**/*')
    files = [x for x in everything if x.is_file() and x.name.endswith('.fts')]

    files.sort(key=get_datetime_from_name)

    for file in files:
        record = Record.find_by_date(
            date_object=get_datetime_from_name(file)
        )

        image, header = read_file(file)

        record.qs_intensity = np.nanmean(
            image[160:330, 723:923]
        )

        record.qs_std = np.nanstd(
            image[160:330, 723:923]
        )

        record.save()


@timeit
def do_all(base_path, write_path, dividor, remainder):
    everything = base_path.glob('**/*')
    files = [x for x in everything if x.is_file() and x.name.endswith('.fts')]

    files.sort(key=get_datetime_from_name)

    for index, file in enumerate(files):
        if index % dividor != remainder:
            continue

        image, header = read_file(file)

        k_list = [
            -3.02,
            -3.05,
            -3.08,
            -3.11,
            -3.14,
            -3.17,
            -3.20,
            -3.23,
            -3.26
        ]

        record = Record(
            date_time=get_datetime_from_name(file)
        )

        record.qs_intensity = np.nanmean(
            image[160:330, 723:923]
        )

        record.qs_std = np.nanstd(
            image[160:330, 723:923]
        )

        record = record.save()

        for k in k_list:

            save_for_this_segment(image, header, k, record, file, write_path)
