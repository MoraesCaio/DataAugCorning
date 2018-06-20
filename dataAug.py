import numpy as np
from dip import ImageMatrix


# sat_tolerance MUST HAVE an equal, or greater, value on generate_sample()'s
#  call than the one used on get_valid_resize_ratios()'s call.
#  Otherwise, it probably won't find the same contours.

def is_valid_resize_ratio(img_mtx, ratio=1.,
                          sat_tolerance=0, verbose=0, show=False):

    resized = img_mtx.resize(ratio=ratio)
    cropped = resized.crop_max_contour()
    median = cropped.get_median_rgb_value()[:3]

    if verbose >= 2:
        print('Checking resized image with ratio == ' +
              "{0:.2f}".format(ratio) + ' Shape' + str(resized.shape))

    # Check if median color is not black
    #  and if the difference between channels is greater the sat_tolerance
    #  (for avoiding nearly monochromatic images)
    if (median != np.array([0, 0, 0])).any() and\
       cropped.is_saturation_greater_than(sat_tolerance):

        if verbose >= 1:
            print('OK: Shape' + str(resized.shape) +
                  ' ratio = ' + "{0:.2f}".format(ratio))

        if show:
            cropped.show('Crop without filling')

        return True

    return False


def get_valid_resize_ratios(img_mtx, sat_tolerance=0,
                            start=.15, stop=.21, step=.005,
                            verbose=0, show=False):

    fl_lst = []

    for i in np.arange(start, stop, step):
        if is_valid_resize_ratio(img_mtx, i, sat_tolerance, verbose, show):
            fl_lst.append(i)

    return fl_lst


def generate_sample(img_mtx, dim, find_max_resize_ratio=True, proportion=0.9,
                    sat_tolerance=0, raw=False, tup2_resize=(0.95, 1.05),
                    tup2_rotation=(-10, 10), tup2_shine=(0.7, 1.05)):

    # Condition to make the function less error prone
    if dim % 4:
        raise Exception('Dimension should be multiple of 4.')

    # Creating black canvas
    background = np.ones((dim, dim, img_mtx.shape[2]), dtype='uint8') * 255
    background[:, :, :3] = 0

    # Resizing beyond 2 times may cause the
    #  image to be greater than the output canvas
    if tup2_resize[1] >= 2.:
        raise Exception('Maximum possible resize factor (' +
                        str(tup2_resize[1]) + ') is error prone.')

    # If img_mtx is not on a valid size ratio search for a new one
    if find_max_resize_ratio and not is_valid_resize_ratio(img_mtx, 1.):

        resize_ratios = get_valid_resize_ratios(img_mtx, sat_tolerance)

        # Could not find a new ratio, returns a black image
        if len(resize_ratios) == 0:
            print('Could not find a valid resize ratio for this image.')
            print('(Returning a black image with specified dimension.)')
            return background.view(ImageMatrix)

        img_mtx = img_mtx.resize(ratio=resize_ratios[-1])

    # Get a valid random resize factor
    min_resize, max_resize = tup2_resize[0], tup2_resize[1]
    rnd_resize_ratio = np.random.uniform(min_resize, max_resize)

    # Get random rotation angle
    min_rotation, max_rotation = tup2_rotation[0], tup2_rotation[1]
    rnd_rotation = int(np.random.uniform(min_rotation, max_rotation + 1))

    # Get random shine factor
    min_shine, max_shine = tup2_shine[0], tup2_shine[1]
    rnd_shine = np.random.uniform(min_shine, max_shine)

    # Applying random rotation
    rnd_rotated = img_mtx.rotate_cropping(
        rnd_rotation) if not raw else img_mtx.crop_max_contour()

    # Adjusting to fit in canvas
    largest = rnd_rotated.shape[1] if rnd_rotated.shape[1] > rnd_rotated.shape[0] else rnd_rotated.shape[0]
    fitting_resize_ratio = dim / (1 / proportion) / largest
    resize_ratio = fitting_resize_ratio * \
        rnd_resize_ratio if not raw else fitting_resize_ratio
    resized = rnd_rotated.resize(ratio=resize_ratio)

    # Finding suitable location to place the resulting image
    x = (dim - resized.shape[1]) // 2
    y = (dim - resized.shape[0]) // 2

    # Applying random shine
    new_shine = resized.multiply_shine(rnd_shine) if not raw else resized

    # Placing resulting image on canvas
    background[y:y + new_shine.shape[0], x:x + new_shine.shape[1]] = new_shine

    return background.view(ImageMatrix)
