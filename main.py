import os
import argparse
import cv2
from dip import ImageMatrix
import dataAug

parser = argparse.ArgumentParser()

parser.add_argument('-f', '--filename', type=str, help='Uses a specific file.')
parser.add_argument('-F', '--folder', type=str, help='Iterates through every file inside a folder.')
parser.add_argument('-d', '--dimension', type=int, default=224, help='Dimension of the output image. Both height and width will be equal to dimension (default value = 224).')
parser.add_argument('-r', '--raw', action='store_true', help='Does not apply random features to image.')
parser.add_argument('-n', '--number-samples', type=int, default=1, help='Number of samples generated (default value = 1).')
parser.add_argument('-s', '--save', nargs='?', const='', help='Saves output image with specified name. If -s is used, but no name is given, output file will be saved as \'prefix_filename.ext\'.')
parser.add_argument('-o', '--output-folder', nargs='?', default='', const='output/', help='Output folder. If -o is used, but no folder is given, a new folder \'output/\' will be created for storing output file.')
parser.add_argument('-p', '--prefix', nargs='?', default='', const='OUT_', help='Prefix added before every output file.')
parser.add_argument('-v', '--view', action='store_true', help='Shows resulting image.')
parser.add_argument('-T', '--top-margin', type=float, default=0.19, help='Top margin (percentual). Crops image from the top.')
parser.add_argument('-B', '--bottom-margin', type=float, default=0.31, help='Bottom margin (percentual). Crops image from the bottom.')
parser.add_argument('-L', '--left-margin', type=float, default=0.31, help='Left margin (percentual). Crops image from the left.')
parser.add_argument('-R', '--right-margin', type=float, default=0.29, help='Right margin (percentual). Crops image from the right.')


args, _ = parser.parse_known_args()

# Called -F
if args.folder:
    folder = args.folder if args.folder.endswith('/') else args.folder + '/'

    # -O
    output_folder = args.output_folder if args.output_folder.endswith('/') or not args.output_folder else args.output_folder + '/'
    os.makedirs(folder + output_folder, exist_ok=True)

    option = 's'
    if args.save == '' and not output_folder and not args.prefix:
        option = input('This will overwrite files. Are you sure? [s/N]\n')

    for filename in os.listdir(folder):
        if filename[-3:] in ['jpg', 'png', 'bmp', 'JPG', 'JPEG']:

            img = ImageMatrix.from_file(folder + filename)
            for i in range(args.number_samples):
                cropped_img = dataAug.crop_margins(img, args.top_margin, args.bottom_margin, args.left_margin, args.right_margin)
                if args.raw:
                    out = dataAug.extract_raw_sample(cropped_img, args.dimension)
                else:
                    out = dataAug.generate_random_sample(cropped_img, args.dimension)

                # -p
                output_path = folder + output_folder + args.prefix + str(i) + filename

                # Called -s
                if args.save is not None:
                    if option == 's':
                        out.save(output_path)

                # Called -v
                if args.view:
                    out.show(filename, wait=False)
                    cv2.waitKey(500)

    # Called -v
    if args.view:
        cv2.waitKey()

else:

    filename = args.filename
    img = ImageMatrix.from_file(filename)
    for i in range(args.number_samples):
        cropped_img = dataAug.crop_margins(img, args.top_margin, args.bottom_margin, args.left_margin, args.right_margin)
        if args.raw:
            out = dataAug.extract_raw_sample(cropped_img, args.dimension)
        else:
            out = dataAug.generate_random_sample(cropped_img, args.dimension)

        # -O
        output_folder = args.output_folder if args.output_folder.endswith('/') or not args.output_folder else args.output_folder + '/'
        if output_folder:
            os.makedirs(output_folder, exist_ok=True)

        # Called -s
        if args.save is not None:
            # Did not pass -s value
            if not args.save:
                option = 's'
                if not output_folder and not args.prefix:
                    option = input('This will overwrite file. Are you sure? [s/N]\n')
                if option == 's':
                    out.save(output_folder + args.prefix + str(i) + filename)
            else:
                out.save(output_folder + args.prefix + str(i) + args.save)

        # Called -v
        if args.view:
            out.show(filename)

print('Finished.')
