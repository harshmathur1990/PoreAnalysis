import pathlib
import datetime
import utils
import numpy as np
import sunpy.io.fits
import sunpy.map
import astropy.units as u
import astropy.coordinates
import sunpy.image.resample
import scipy.ndimage
import matplotlib.pyplot as plt
from sunpy.image.coalignment \
    import calculate_match_template_shift,\
    mapsequence_coalign_by_match_template as mc_coalign


def compare_two_images(image1, image2):
    fig, axs = plt.subplots(2)

    axs[0].imshow(image1, cmap='gray')

    axs[1].imshow(image2, cmap='gray')

    plt.show()


def do_align_hmi_with_hifi(hifi_path, hmi_image):
    hifi_data, hifi_header = sunpy.io.fits.read(hifi_path)[0]

    hmi_data, hmi_header = sunpy.io.fits.read(hmi_image)[1]

    hmi_map = sunpy.map.Map(hmi_data, hmi_header)

    corrected_hmi_map = hmi_map.rotate()

    # spread = 58.995

    spread = 29.4975

    init = (-499.1 - spread / 2, -295 - spread / 2)

    final = (-499.1 + spread / 2, -295 + spread / 2)

    y0 = init[1] * u.arcsec

    x0 = init[0] * u.arcsec

    xf = final[0] * u.arcsec

    yf = final[1] * u.arcsec

    bottom_left1 = astropy.coordinates.SkyCoord(
        x0, y0, frame=hmi_map.coordinate_frame
    )

    top_right1 = astropy.coordinates.SkyCoord(
        xf, yf, frame=hmi_map.coordinate_frame
    )

    submap = corrected_hmi_map.submap(bottom_left1, top_right1)

    resampled_hmi_image = sunpy.image.resample.resample(
        orig=submap.data,
        dimensions=(
            submap.data.shape[0] * submap.meta['cdelt1'] / 0.0253,
            submap.data.shape[1] * submap.meta['cdelt2'] / 0.0253
        ),
        method='spline',
        minusone=False
    )

    new_meta = submap.meta.copy()

    new_meta['naxis1'] = resampled_hmi_image.shape[0]
    new_meta['naxis2'] = resampled_hmi_image.shape[1]
    new_meta['cdelt1'] = 0.0253
    new_meta['cdelt2'] = 0.0253

    new_submap = sunpy.map.Map(resampled_hmi_image, new_meta)

    angle = -20

    rotated_hifi_data = scipy.ndimage.rotate(
        hifi_data,
        angle=angle,
        order=3,
        prefilter=False,
        reshape=False,
        cval=np.nan
    )

    rotated_hifi_data[np.where(np.isnan(rotated_hifi_data))] = 0.0

    rotated_meta = new_meta.copy()

    rotated_meta['naxis1'] = rotated_hifi_data.shape[0]

    rotated_meta['naxis2'] = rotated_hifi_data.shape[1]

    rotated_map = sunpy.map.Map(rotated_hifi_data, rotated_meta)

    map_sequence = sunpy.map.Map((rotated_map, new_submap), sequence=True)

    return calculate_match_template_shift(
        map_sequence
    ), hifi_data, rotated_hifi_data, new_submap


def get_datetime_from_hifi_image(file):
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


def prepare_get_corresponding_images(base_path):

    everything = base_path.glob('**/*')
    vis_images = [
        x for x in everything if x.is_file() and x.name.endswith('.fits')
    ]

    vis_ordered_list = list()

    for vis_image in vis_images:
        vis_ordered_list.append(utils.get_hmi_jul_day(vis_image))

    vis_ordered_list = np.array(vis_ordered_list)

    def get_corresponding_images(hifi_image):

        hifi_date = get_datetime_from_hifi_image(hifi_image)

        hifi_julian_day = utils.get_julian_time(hifi_date)

        vis_subtract_array = np.abs(vis_ordered_list - hifi_julian_day)

        vis_argmin = np.argmin(vis_subtract_array)

        if vis_subtract_array[vis_argmin] < (45 / 86400):
            return vis_images[vis_argmin], True

        return None, False

    return get_corresponding_images


def call_align_func(hifi_path, base_path_hmi):
    get_corresponding_images = prepare_get_corresponding_images(base_path_hmi)

    hmi_image, status = get_corresponding_images(hifi_path)

    return do_align_hmi_with_hifi(hifi_path, hmi_image)


def call_one():

    hifi_path = pathlib.Path(
        '/Volumes/Harsh 9599771751/Gregor Data/' +
        'filtrd1/filtrd_hifi_20170928_085120_sd_speckle.fts'
    )

    base_path_hmi = pathlib.Path(
        '/Volumes/Harsh 9599771751/Continuum'
    )

    return call_align_func(hifi_path, base_path_hmi)
