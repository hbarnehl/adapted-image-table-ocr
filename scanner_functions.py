# Importing other necessary packages
import glob
import pandas as pd
import os
import concurrent # for parallel instances
import functools # for creating partial functions
import shutil
from pathlib import Path
from io import StringIO # to convert string to csv

# to add the path where to search for modules
import sys
sys.path.append('/home/hennes/Internship/table_scanner')

# Importing table_ocr modules 
from table_ocr import pdf_to_images
from table_ocr import extract_tables
from table_ocr import extract_cells
from table_ocr import ocr_image
from table_ocr import ocr_to_csv

## setting folders and pdfs
folder = "/home/hennes/Internship/pdfs/" # should be folder containing pdfs of election
save_folder = '/home/hennes/Internship/constituencies/' # folder into which csvs should be saved
saved = [os.path.splitext(csv)[0] for csv in next(os.walk(save_folder))[2]]
old = '/home/hennes/Internship/old_files/' # folder into which old files are moved
old_files = [folder for folder in next(os.walk(old))[1]]
allpdf = [pdf for pdf in glob.glob(folder+'*') if pdf.endswith(".pdf")] # list with all pdfs from folder

def move_files(folder, old):
    # if folder already exist, delete it. Then move folders to old directory.
    [shutil.rmtree(old+'/'+e)
     for e in next(os.walk(folder))[1] if Path(old+'/'+e).is_dir()]

    [shutil.move(folder+''+e, old)
     for e in next(os.walk(folder))[1] if not e.endswith('.pdf')]

    # if files already exist, delete them. Then move files to old directory.
    [os.remove(old+'/'+e)
     for e in next(os.walk(folder))[2] if not e.endswith('.pdf') and Path(old+'/'+e).is_file()]

    [shutil.move(e, old)
     for e in glob.glob(folder+'*') if not e.endswith('.pdf')]

def ocr_pipeline(pdf, thresh=None, no_noise=None, preprocess=True, dilate=True, image_conversion=True,
                 AC = False, from_cell = False):
    '''Function which binds together the functions necessary to turn one pdf of several pages of tables
    into a csv file which will be stored under the same name in the specified folder.
    The options are:
    
    image_conversion - whether images should be converted from pdf. If value is None, then converted 
                       images should already be supplied in the folder.
    
    thresh           - whether otsu thresholding should be applied to cell images before performing ocr.
                       Useful if the images contain a lot of grey, in which case not thresholding likely
                       results in many artifacts wrongly identified as numbers.
    
    no_noise         - whether noise reduction techniques should be used before ocr. It should be avoided in
                       images with very thin/small font. Helpful in case of cropping error (in that case, put TRUE).
                
    preprocess       - whether preprocessing should be used before table extraction. Useful to avoid if
                       preprocessing results in wrongly rotated images. Can only be used if function is
                       used on already extracted images.
                 
    dilate           - whether image should be dilated before cell extraction. Useful for thin/frail cell
                       lines, but results in numbers becoming so thick that they are recognised as cell 
                       walls themselves in images with tightly written fonts.
    
    AC               - whether the folders created with table_extraction should be named after the AC con-
                       stituency, which is extracted from the images using OCR. This is necessary for the 
                       national elections, which use a different constituency system. Pipeline stops after
                       the renaming.
                       
    from_cell        - "True" = workflow starts from cell extraction. Only works if input is folders from 
                       old folder.
                       "False" = workflow starts from image conversion. Input are pdf files.'''
    
    stop = None
    
    if from_cell == False:
        if image_conversion == True:
            # extract pages from pdf and save as images 
            pdf_to_images.pdf_to_images(pdf)
            print(f"created images of {pdf.split('/')[-1]}")

        # define list of images thus created
        imglist = [img for img in glob.glob(folder+'*') if img.endswith('.png')]

        if preprocess == True:
            # rotate and correct images for skew
            with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
                executor.map(pdf_to_images.preprocess_img, imglist)
            print(f"preprocessed images of {pdf.split('/')[-1]}")

        # crop images to table and save in new folder
        if AC == False:
            # define imglist as list of lists because next function needs list as argument
            imglist = [[img] for img in imglist]

            # Create partial function for table extraction in which AC is negative
            p_extract_tables = functools.partial(extract_tables.main, AC)

            with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
                executor.map(p_extract_tables, imglist)
            print(f"extracted tables of {pdf.split('/')[-1]}")

        elif AC == True:
            # sort imglist
            imglist = sorted(imglist)

            extract_tables.main(AC, imglist)
            print(f"extracted tables of {pdf.split('/')[-1]} and renamed them.")
            move_files(folder, old)
            return stop

    # define list of all images of tables in newly created subfolders
    dirlist = sorted([directory for directory in glob.glob(folder+'*/*') if directory.endswith('.png')])
    
    # If one of the images is smaller than 50 KB, table extraction probably did not work.
    # In that case, stop and continue with next pdf.
    
    if any(e for e in dirlist if os.path.getsize(e)/1000 < 50):
        print(f'problem tables: {[e.split("/")[-2] for e in dirlist if os.path.getsize(e)/1000 < 50]}')
        print(f'Table extraction did not work correctly. Continuing with next pdf.')
        move_files(folder, old)
        return stop
        
    # Create partial function for cell extraction in which dilation is specified
    p_extract_cells = functools.partial(extract_cells.main, dilate)

    # Extract individual cell images
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        executor.map(p_extract_cells, dirlist)
    print(f"extracted cells of {pdf.split('/')[-1]}")

    # define list of directories containing cell images
    dirlist = [directory for directory in glob.glob(folder+'*/*/')]

    # define list of images of cells within directories
    celllists = [glob.glob(cellfolder+'*') for cellfolder in dirlist]

    # Specify that there should be no multiple threads.
    # This is important because I am already using multiple processors.
    # And create partial function to pre-specify thresh and no_noise options.
    os.environ['OMP_THREAD_LIMIT'] = '1'
    p_ocr_image = functools.partial(ocr_image.main, None, thresh, no_noise)

    # perform OCR on each image
    for image_list in celllists:
        with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
            executor.map(p_ocr_image, image_list)
    print(f"completed ocr of {pdf.split('/')[-1]}")

    try:
        gathered_data = []
        # get the names for the individual pages
        pages = sorted([filename for directory, filename in
                 [os.path.split(x) for x in glob.glob(folder+'*') if not x.endswith(('.pdf', '.png'))]])

        # create list of alphabetically ordered lists of ocred files
        ocrlists = [sorted(y) for y in
                    [glob.glob(f'{folder}/{x}/cells/ocr_data/*.txt') for x in pages]]
        zippie = zip(pages, ocrlists)

        # for each pair of page and ocred cells, create a csv
        for y, x in zippie:
            output = ocr_to_csv.main(x)
            csv = StringIO(output)
            print(f'working on {y}')

            # Turning csv into dataframe
            # Skipping the first two rows because they have fewer columns than rest
            # Also useful for chaining of tables later
            df = pd.read_csv(csv,  header = None, skiprows=[0, 1])
            gathered_data.append(df)
            df = pd.concat(gathered_data)

            # give df a name and save it
            if from_cell == False:
                constituency_name = pages[0][0:5]
            else:
                constituency_name = pages[0].split('-')[1]
            df.to_csv(save_folder+constituency_name+'.csv')

        print(f'Saved {constituency_name} to folder.')

        # move old files and folders into old_files folder
        move_files(folder, old)

    # If there is an error, print error message.
    # Most likely eror is that header lines (which are not turned into cells properly)
    # are three instead of two rows. So we try reading the csvs again, this
    # time ignoring the first three instead of just two rows.
    except Exception as e:
        print(e)
        print('will try again ignoring one more line.')
            # get the names for the individual pages

        try:
            gathered_data = []
            pages = sorted([filename for directory, filename in
                     [os.path.split(x) for x in glob.glob(folder+'*') if not x.endswith(('.pdf', '.png'))]])

            # create list of alphabetically ordered lists of ocred files

            ocrlists = [sorted(y) for y in
                        [glob.glob(f'{folder}/{x}/cells/ocr_data/*.txt') for x in pages]]
            zippie = zip(pages, ocrlists)

            # for each pair of directory and files, create csv file
            for y, x in zippie:
                output = ocr_to_csv.main(x)
                csv = StringIO(output)
                print(f'working on {y}')

                # Turning csv files into single dataframe
                # Skipping the first two rows because they have fewer columns than rest
                # Also useful for chaining of tables later
                col_names = [str(e) for e in list(range(0,25))]
                df = pd.read_csv(csv,  header = None, skiprows=[0, 1], names=col_names)
                gathered_data.append(df)
                df = pd.concat(gathered_data)

                # give df a name and save it
                if from_cell == False:
                    constituency_name = pages[0][0:5]
                else:
                    constituency_name = pages[0].split('-')[1]
                df.to_csv(save_folder+constituency_name+'.csv')

            print(f'Saved {constituency_name} to folder.')

            move_files(folder, old)

        # If this still does not work, stop the program and try to manually correct mistakes.
        except Exception as e:
            print(e)
            print(f'There is a problem with {pdf.split("/")[-1]}. Continuing with next pdf.')
            # move old files and folders into old_files folder
            move_files(folder, old)