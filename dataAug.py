import numpy as np
from dip import ImageMatrix


# sat_tolerance MUST HAVE an equal, or greater, value on generate_sample()'s
#  call than the one used on get_valid_resize_ratio()'s call.
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


def get_valid_resize_ratio(img_mtx, sat_tolerance=0,
                           start=.2, stop=.01, step=-.005,
                           verbose=0, show=False):

    for resize_ratio in np.arange(start, stop, step):
        if is_valid_resize_ratio(img_mtx, resize_ratio, sat_tolerance,
                                 verbose, show):
            return resize_ratio


def generate_random_sample(img_mtx, dim, proportion=0.9,
                           sat_tolerance=0, tup2_resize=(0.95, 1.05),
                           tup2_rotation=(-10, 10), tup2_shine=(0.7, 1.05)):

    # Condition to make the function less error prone
    if dim % 4:
        raise Exception('Dimension should be multiple of 4.')

    # Creating black canvas
    background = np.zeros((dim, dim, img_mtx.shape[2]), dtype='uint8')
    background[:, :, 3:] = 255

    # Resizing beyond 2 times may cause the
    #  image to be greater than the output canvas
    if tup2_resize[1] >= 2.:
        raise Exception('Maximum possible resize factor (' +
                        str(tup2_resize[1]) + ') is error prone.')

    # If img_mtx is not on a valid size ratio search for a new one
    if not is_valid_resize_ratio(img_mtx, 1.):

        resize_ratio = get_valid_resize_ratio(img_mtx, sat_tolerance)

        # Could not find a new ratio, returns a black image
        if resize_ratio < 0.02:
            print('Could not find a valid resize ratio for this image.')
            print('(Returning a black image with specified dimension.)')
            return background.view(ImageMatrix)

        img_mtx = img_mtx.resize(ratio=resize_ratio)

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
    rnd_rotated = img_mtx.rotate_cropping(rnd_rotation)

    # Adjusting to fit in canvas
    largest = max(rnd_rotated.shape[1], rnd_rotated.shape[0])
    fitting_resize_ratio = dim * proportion / largest
    resize_ratio = fitting_resize_ratio * rnd_resize_ratio
    resized = rnd_rotated.resize(ratio=resize_ratio)

    # Finding suitable location to place the resulting image
    x = (dim - resized.shape[1]) // 2
    y = (dim - resized.shape[0]) // 2

    # Applying random shine
    new_shine = resized.multiply_shine(rnd_shine)

    # Placing resulting image on canvas
    background[y:y + new_shine.shape[0], x:x + new_shine.shape[1]] = new_shine

    return background.view(ImageMatrix)


def extract_raw_sample(img_mtx, dim, proportion=0.9, sat_tolerance=0):

    # Creating black canvas
    background = np.zeros((dim, dim, img_mtx.shape[2]), dtype='uint8')
    background[:, :, 3:] = 255

    # If img_mtx is not on a valid size ratio search for a new one
    if not is_valid_resize_ratio(img_mtx, 1.):

        resize_ratio = get_valid_resize_ratio(img_mtx, sat_tolerance)

        # Could not find a new ratio, returns a black image
        if resize_ratio < 0.02:
            print('Could not find a valid resize ratio for this image.')
            print('(Returning a black image with specified dimension.)')
            return background.view(ImageMatrix)

        img_mtx = img_mtx.resize(ratio=resize_ratio)

    # Applying random rotation
    max_contour = img_mtx.crop_max_contour()

    # Adjusting to fit in canvas
    largest = max(max_contour.shape[1], max_contour.shape[0])
    resize_ratio = dim * proportion / largest
    resized = max_contour.resize(ratio=resize_ratio)

    # Finding suitable location to place the resulting image
    x = (dim - resized.shape[1]) // 2
    y = (dim - resized.shape[0]) // 2

    # Placing resulting image on canvas
    background[y:y + resized.shape[0], x:x + resized.shape[1]] = resized

    return background.view(ImageMatrix)


def crop_margins(img_mtx, top_margin, bottom_margin, left_margin, right_margin):

    # print('height: ' + str(img_mtx.shape[0]) +
    #       '\nwidth: ' + str(img_mtx.shape[1]))
    
    top    = int(img_mtx.shape[0] * top_margin)
    bottom = int(img_mtx.shape[0] - img_mtx.shape[0] * bottom_margin)
    left   = int(img_mtx.shape[1] * left_margin)
    right  = int(img_mtx.shape[1] - img_mtx.shape[1] * right_margin)

    # img_mtx[top:bottom, left:right].view(ImageMatrix).get_image().show()
    # img_mtx[450:1600, 1300:2900].view(ImageMatrix).get_image().show()
    cropped = img_mtx[top:bottom, left:right].view(ImageMatrix)
    cropped.save('0margins.jpg', 'JPEG')
    return cropped
