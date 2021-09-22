# will use file to assemble pipeline

# Preprocessing Table
pdf_to_images.preprocess_img('trial.png')

# Extracting Table Image from PDF Page image
extract_tables.main(['trial.png'])

# perform OCR on each image
for image in [x for x in glob.glob('/home/hennes/Internship/trial/cells/*') if 
x.endswith('.png')]:
    ocr_image.main(image, None)
# have to give 'None' as argument, because not executed in shell script

# Put OCRed str into csv
files = [x for x in glob.glob('/home/hennes/Internship/trial/cells/ocr_data/*') if 
x.endswith('.txt')] files.sort() # files need to be alphabetically sorted output = 
ocr_to_csv.main(files)
csv = StringIO(output)

# Turning csv into dataframe Skipping the first two rows because they have fewer 
# columns than rest Also useful for chaining of tables later
df = pd.read_csv(csv,  header = None, skiprows=[0, 1])
