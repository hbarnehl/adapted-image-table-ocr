# Importing basic packages
import glob
import pandas as pd
import numpy as np
import cv2 # image transformation

from io import StringIO # to convert string to csv
import time # to measure time

# to add the path where to search for modules
import sys
sys.path.append('/home/hennes/Internship/table_scanner')

# set number of simultaneous threads for tesseract
os.environ['OMP_THREAD_LIMIT'] = '1'

# Importing table_ocr modules 
from table_ocr import pdf_to_images
from table_ocr import extract_tables
from table_ocr import extract_cells
from table_ocr import ocr_image
from table_ocr import ocr_to_csv


# perform OCR on each image
def main():
	with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
            	image_list = glob.glob(path+"\\*.png")
            	for img_path,out_file in zip(image_list,executor.map(ocr,image_list)):
                	print(img_path.split("\\")[-1],',',out_file,', processed')
                	
                	
	
	for image in [x for x in glob.glob('/home/hennes/Internship/trial/cells/*') if x.endswith('.png')]:
    	ocr_image.main(image, None)
# have to give 'None' as argument, because not executed in shell script

if __name__ == '__main__':
    main()
