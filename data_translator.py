# importing csv module
import csv
import numpy
from scipy import stats
from googletrans import Translator

from google_trans_new import google_translator


# csv file name
filename = "/home/blin/Downloads/train1.csv"
filename_en = "/home/blin/Downloads/train1en.csv"

# initializing the titles and rows list
fields = []
rows = []

# reading csv file
translator = google_translator(proxies={ 'https':'35.200.61.59:3124', 'https':'201.220.140.30:8181'})
with open(filename) as f_in, open(filename_en, 'w') as f_out:
    reader = csv.reader(f_in, delimiter=',')
    writer = csv.writer(f_out, delimiter=',')
    header = next(reader)
    writer.writerow(header)
    for row in reader:
        translated_text = ""
        if (row[17] != ""):
            start_desc = row[17].find(">")
            end_desc = row[17].find("<")
            text_desc = row[17][start_desc + 1: end_desc]
            print(text_desc)
            print('\n')
            translated_text = translator.translate(text_desc, lang_tgt='en')
            print(translated_text)
            row[17] = translated_text
            print('\n')
            print(row[0])
            print('\n')
        writer.writerow(row)
        # for col in row:
        #     colValues.append(col.lower())
        # writer.writerow(colValues)





