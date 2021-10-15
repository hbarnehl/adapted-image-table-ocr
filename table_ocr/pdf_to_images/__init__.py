import os
import re
import subprocess
from scipy.ndimage import interpolation as inter
import numpy as np
import cv2

from table_ocr.util import get_logger, working_dir

logger = get_logger(__name__)

# Wrapper around the Poppler command line utility "pdfimages" and helpers for
# finding the output files of that command.
def pdf_to_images(pdf_filepath):
    """
    Turn a pdf into images
    Returns the filenames of the created images sorted lexicographically.
    """
    directory, filename = os.path.split(pdf_filepath)
    image_filenames = pdfimages(pdf_filepath)

    # Since pdfimages creates a number of files named each for there page number
    # and doesn't return us the list that it created
    return sorted([os.path.join(directory, f) for f in image_filenames])


def pdfimages(pdf_filepath):
    """
    Uses the `pdfimages` utility from Poppler
    (https://poppler.freedesktop.org/). Creates images out of each page. Images
    are prefixed by their name sans extension and suffixed by their page number.

    This should work up to pdfs with 999 pages since find matching files in dir
    uses 3 digits in its regex.
    """
    directory, filename = os.path.split(pdf_filepath)
    if not os.path.isabs(directory):
        directory = os.path.abspath(directory)
    filename_sans_ext = filename.split(".pdf")[0]

    # pdfimages outputs results to the current working directory
    with working_dir(directory):
        subprocess.run(["pdfimages", "-png", filename, filename.split(".pdf")[0]])

    image_filenames = find_matching_files_in_dir(filename_sans_ext, directory)
    logger.debug(
        "Converted {} into files:\n{}".format(pdf_filepath, "\n".join(image_filenames))
    )
    return image_filenames


def find_matching_files_in_dir(file_prefix, directory):
    files = [
        filename
        for filename in os.listdir(directory)
        if re.match(r"{}-\d{{3}}.*\.png".format(re.escape(file_prefix)), filename)
    ]
    return files

def preprocess_img(filepath, tess_params=None):
    """Processing that involves running shell executables,
    like mogrify to rotate.

    Uses tesseract to detect rotation.

    Orientation and script detection is only available for legacy tesseract
    (--oem 0). Some versions of tesseract will segfault if you let it run OSD
    with the default oem (3).
    """
    if tess_params is None:
        tess_params = ["--psm", "0", "--oem", "0"]
    rotate = get_rotate(filepath, tess_params)
    logger.debug("Rotating {} by {}.".format(filepath, rotate))
    mogrify(filepath, rotate)
    correct_skew(filepath)

def get_rotate(image_filepath, tess_params):
    """
    """
    tess_command = ["tesseract"] + tess_params + [image_filepath, "-"]
    output = (
        subprocess.check_output(tess_command)
        .decode("utf-8")
        .split("\n")
    )
    output = next(l for l in output if "Rotate: " in l)
    output = output.split(": ")[1]
    return output


def mogrify(image_filepath, rotate):
    subprocess.run(["mogrify", "-rotate", rotate, image_filepath])
    
def correct_skew(filepath, delta=1, limit=5):
	'''This is code taken from https://stackoverflow.com/a/57965160 to auto
	matically detect skew in the image of a table and apply the appropriate
	rotation to straigthen it.'''
	def determine_score(arr, angle):
		data = inter.rotate(arr, angle, reshape=False, order=0)
		histogram = np.sum(data, axis=1)
		score = np.sum((histogram[1:] - histogram[:-1]) ** 2)
		return histogram, score

	image = cv2.imread((filepath), cv2.IMREAD_GRAYSCALE)
	thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1] 
	
	scores = []
	angles = np.arange(-limit, limit + delta, delta)
	for angle in angles:
		histogram, score = determine_score(thresh, angle)
		scores.append(score)
	
	best_angle = angles[scores.index(max(scores))]
	
	(h, w) = image.shape[:2]
	center = (w // 2, h // 2)
	M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
	rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
	cv2.imwrite(filepath, rotated)
