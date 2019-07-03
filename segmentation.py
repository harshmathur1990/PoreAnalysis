import sys
import time
import numpy as np
import scipy.signal
import scipy.ndimage
import sunpy.io.fits


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
