import os
import cv2
import pytesseract
import re

def find_tables(image, AC, old, SCALE):
    BLUR_KERNEL_SIZE = (17, 17)
    STD_DEV_X_DIRECTION = 0
    STD_DEV_Y_DIRECTION = 0
    blurred = cv2.GaussianBlur(image, BLUR_KERNEL_SIZE, STD_DEV_X_DIRECTION, STD_DEV_Y_DIRECTION)
    MAX_COLOR_VAL = 255
    BLOCK_SIZE = 15
    SUBTRACT_FROM_MEAN = -2
    
    img_bin = cv2.adaptiveThreshold(
        ~blurred,
        MAX_COLOR_VAL,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        BLOCK_SIZE,
        SUBTRACT_FROM_MEAN,
    )
    vertical = horizontal = img_bin.copy()
    image_width, image_height = horizontal.shape
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(image_width / SCALE), 1))
    horizontally_opened = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, horizontal_kernel)
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(image_height / SCALE)))
    vertically_opened = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, vertical_kernel)
    
    horizontally_dilated = cv2.dilate(horizontally_opened, cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1)))
    vertically_dilated = cv2.dilate(vertically_opened, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 60)))
    
    mask = horizontally_dilated + vertically_dilated
    contours, heirarchy = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )

    MIN_TABLE_AREA = 1e5
    contours = [c for c in contours if cv2.contourArea(c) > MIN_TABLE_AREA]
    perimeter_lengths = [cv2.arcLength(c, True) for c in contours]
    epsilons = [0.1 * p for p in perimeter_lengths]
    approx_polys = [cv2.approxPolyDP(c, e, True) for c, e in zip(contours, epsilons)]
    bounding_rects = [cv2.boundingRect(a) for a in approx_polys]
    
    # If AC option is chosen, then scan the image and search the scanned text for 
    # 'segment'. The first numbers after segment constitute the AC constituency 
    # number. There are three scenarios. If segment is found 2 times, then the
    # constituency number will be the first digits found after the second segment.
    # If segment is found only once, the format is different, and the AC number will
    # be written behind 'constituency'. If there is no 'segment', then use the same 
    # AC number as from the former document. 
    
    if AC == True:
        try:
            nimg = image
            txt = pytesseract.image_to_string(nimg)
            if len(re.findall(r'[Ss]e[gq]ment', txt)) >= 2:
                txt = re.search(r'[Ss]e[gq]ment\s?[.:-]?\s*?\d{1,3}\s?\-?\s?[a-zA-Z]', txt).group(0)
                txt = re.search(r'\d{1,3}', txt).group(0)
            elif len(re.findall(r'[Ss]e[gq]ment', txt)) == 1:
                txt = re.search(r'[Cc]onstituency\s?[.:-]?\s?\d{1,3}\s?\-?\s?[a-zA-Z]', txt).group(0)
                txt = re.search(r'\d{1,3}', txt).group(0)
            elif len(re.findall(r'[Ss]e[gq]ment', txt)) == 0:
                txt = re.search(r'Part I\s*?\d{1,3}\s?\-\s?[a-zA-Z]', txt).group(0)
                txt = re.search(r'\d{1,3}', txt).group(0)
        except:
            txt = old
            None
    elif AC == False:
        txt = None

    # The link where a lot of this code was borrowed from recommends an
    # additional step to check the number of "joints" inside this bounding rectangle.
    # A table should have a lot of intersections. We might have a rectangular image
    # here though which would only have 4 intersections, 1 at each corner.
    # Leaving that step as a future TODO if it is ever necessary.
    images = [image[y:5+y+h, x-10:x+w+10] for x, y, w, h in bounding_rects]
    try:
        if images[0].size == 0 or len(images[0]) < 50:
            images = [image[y:5+y+h, x-5:x+w+5] for x, y, w, h in bounding_rects]
            if images[0].size == 0 or len(images[0]) < 50:
                images = [image[y:5+y+h, x:x+w+5] for x, y, w, h in bounding_rects]
                if images[0].size == 0 or len(images[0]) < 50:
                    images = [image[y:y+h, x:x+w] for x, y, w, h in bounding_rects]
    except:
        images = None 
        	
    return images, txt

def main(AC, files):
    results = []
    old = None
    for f in files:
        directory, filename = os.path.split(f)
        filename_sans_extension = os.path.splitext(filename)[0]
        image = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        tables, txt = find_tables(image, AC, old, SCALE=8)
        if AC:
            if txt == None:
                print(f'{filename_sans_extension}: no AC number found')
            else:
                print(f'{filename_sans_extension}: AC = {txt}')
            old = txt
        if tables == None or tables[0].size == 0 or len(tables[0]) < 50:
            tables, txt = find_tables(image, AC, old, SCALE=20)
            if AC:
                old = txt
            if tables == None or tables[0].size == 0 or len(tables[0]) < 50:
                print(f'Extraction error: {f}.')
        files = []
        
        # If AC option was chosen, folder name will be AC instead of the constituency of the 
        # pdf.
        if AC == True:
            if txt == None:
                txt = '000'
            else:
                txt = "{:03d}".format(int(txt))
            PC = filename_sans_extension.split('-')[0]
            filename_sans_extension = f'{PC}-AC{txt}-{filename_sans_extension.split("-")[-1]}'
        if tables:
            os.makedirs(os.path.join(directory, filename_sans_extension), exist_ok=True)
            for i, table in enumerate(tables):
                table_filename = "table-{:03d}.png".format(i)
                table_filepath = os.path.join(
                    directory, filename_sans_extension, table_filename)
                files.append(table_filepath)
                cv2.imwrite(table_filepath, table)
            results.append((f, files))
    # Results is [[<input image>, [<images of detected tables>]]]
    return results
