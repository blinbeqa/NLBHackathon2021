# importing csv module
import csv
import numpy
from scipy import stats


# csv file name
filename = "/home/blin/Downloads/train1en.csv"

# initializing the titles and rows list
fields = []
rows = []

cluster_labels = numpy.load("/home/blin/PycharmProjects/nlb/cluster_label.npy")

# reading csv file
with open(filename, 'r') as csvfile:
    # creating a csv reader object
    csvreader = csv.reader(csvfile)

    # extracting field names through first row
    fields = next(csvreader)

    # extracting each data row one by one
    for row in csvreader:
        rows.append(row)

    # get total number of rows
    print("Total no. of rows: %d" % (csvreader.line_num))

# printing the field names
print('Field names are:' + ', '.join(field for field in fields))

#  printing first 5 rows
# temp_list = []
# print('\nFirst 5 rows are:\n')
# for i in range(3, len(fields)):
#     print(fields[i])
#     print('\n')
#     for row in rows:
#         #print("%10s" % row[1]),
#         #print('\n')
#         # parsing each column of a row
#         # for col in row:
#         #     print("%10s" % col),
#         # print('\n')
#
#
#         if row[i] not in temp_list:
#             temp_list.append(row[i])
#
#
#     print('\n')
#     print(temp_list)
#     temp_list = []

all_rows = []
all_rows_valid = []
cl_count = 0
counter_valid = 0
yes = 0
no = 0
count_yes = 0
for row in rows:
    temp_row = []
    # LoanID
    #temp_row.append(row[1])
    # ApplicantID
    #temp_row.append(row[2])
    # ApplicantGender
    gender = 0
    if row[3] == "Moški":
        gender = 1
    elif row[3] == "Ženska":
        gender = 2
    temp_row.append(gender)
    # ApplicantMarried
    marr = 0
    if row[4] == "DA":
        marr = 1
    elif row[4] == "NE":
        marr = 2
    temp_row.append(marr)
    # ApplicantDependents
    dep = 0
    if row[5] == "0 oseb":
        dep = 1
    elif row[5] == "1 oseba":
        dep = 2
    elif row[5] == "2 osebi":
        dep = 3
    elif row[5] == "3+ osebe":
        dep = 4
    temp_row.append(dep)
    # ApplicantEducation
    edu = 0
    if row[6] == "Diplomiral":
        edu = 1
    elif row[6] == "Brez diplome":
        edu = 2
    temp_row.append(edu)
    # ApplicantSelfEmployed
    se = 0
    if row[7] == "DA":
        se = 1
    elif row[7] == "NE":
        se = 2
    temp_row.append(se)
    # ApplicantIncome
    income = 0
    if row[8] == "":
        income = 0
    else:
        income = row[8]
    temp_row.append(income)
    # ApplicantCreditHistory
    credhis = 0
    if row[9] == "0.0":
        credhis = 1
    elif row[9] == "1.0":
        credhis = 2
    temp_row.append(credhis)
    # ApplicantZIP
    zipc = 0
    if row[10] == "":
        zipc = 0
    else:
        zipc = row[10][:-2]
    temp_row.append(zipc)
    # ApplicantState
    astate = 1
    temp_row.append(astate)
    # ApplicantEmplLength
    empl = 0
    if row[12] == "< 1 leto":
        empl = 1
    elif row[12] == "1 leto":
        empl = 2
    elif row[12] == "2 leti":
        empl = 3
    elif row[12] == "3 leta":
        empl = 4
    elif row[12] == "4 leta":
        empl = 5
    elif row[12] == "5 let":
        empl = 6
    elif row[12] == "6 let":
        empl = 7
    elif row[12] == "7 let":
        empl = 8
    elif row[12] == "8 let":
        empl = 9
    elif row[12] == "9 let":
        empl = 10
    elif row[12] == "10+ let":
        empl = 11

    temp_row.append(empl)
    # ApplicantHomeOwn
    ho = 0
    if row[13] == "STANOVANJSKI KREDIT":
        ho = 1
    elif row[13] == "NAJEM":
        ho = 2
    elif row[13] == "LASTNO":
        ho = 3
    temp_row.append(ho)
    # LoanAmount
    loa = 0
    if row[14] == "":
        loa = 0
    else:
        loa = row[14]
    temp_row.append(loa)
    # LoanTerm
    lot = 0
    if row[15] == "":
        lot = 0
    else:
        lot = row[15]
    temp_row.append(lot)
    # LoanIntRate
    loir = 0
    if row[16] == "":
        loir = 0
    else:
        loir = row[16][:-1]
    temp_row.append(loir)
    # Loandesctext
    ldt = 0
    if row[17] == "":
        ldt = 0
    else:
        ldt = cluster_labels[cl_count] + 1
        cl_count+=1
        print(cl_count)
    #temp_row.append(ldt)

    # LoanPurpose
    lp = 0
    if row[18] == "prenova":
        lp = 1
    elif row[18] == "drugo":
        lp = 2
    elif row[18] == "kartica":
        lp = 3
    elif row[18] == "zdravljenje":
        lp = 4
    elif row[18] == "poèitnice":
        lp = 5
    elif row[18] == "selitev":
        lp = 6
    elif row[18] == "investicija":
        lp = 7
    elif row[18] == "stanovanje":
        lp = 8
    elif row[18] == "obnovljivi_viri":
        lp = 9
    temp_row.append(lp)
    # LoanApproved
    lapr = 0
    if row[19] == "Y":
        lapr = 1
        yes+=1
    else:
        no+=1
    temp_row.append(lapr)

    if counter_valid % 5 == 0:

        all_rows_valid.append(temp_row)
    else:
        # if row[19] == "Y":
        #     if count_yes % 3 == 0:
        #         all_rows.append(temp_row)
        #     count_yes+=1
        # else:
        all_rows.append(temp_row)
    counter_valid+=1

    # if(row[17] != ""):
    #
    #     text_desc = row[17]
    #
    #     print('\n')
    #     print(text_desc)
    #     print('\n')

print('\n')
print("all_rows_new_format:")
print('\n')
all_rows_npy=numpy.array([numpy.array(xi) for xi in all_rows]).astype(float)
all_rows_valid_npy=numpy.array([numpy.array(xi) for xi in all_rows_valid]).astype(float)
print(all_rows_npy.shape[1])
#print(all_rows_npy[55048])
for i in range(all_rows_npy.shape[1]):
    all_rows_npy[:, i: i + 1] = all_rows_npy[:, i: i+1] / all_rows_npy[:, i: i+1].max()
    print(all_rows_npy[:, i: i+1].max())

for i in range(all_rows_valid_npy.shape[1]):
    all_rows_valid_npy[:, i: i + 1] = all_rows_valid_npy[:, i: i+1] / all_rows_valid_npy[:, i: i+1].max()
    print(all_rows_valid_npy[:, i: i+1].max())


z = numpy.abs(stats.zscore(all_rows_npy))

threshold = 3
indices_to_remove = numpy.unique(numpy.where(z > threshold)[0])
print(indices_to_remove)
print(len(indices_to_remove))
print("length: ", len(all_rows_npy))
#all_rows_npy = numpy.delete(all_rows_npy, indices_to_remove, 0)
print("length: ", len(all_rows_valid_npy))
numpy.save("/home/blin/PycharmProjects/nlb/train_split_train1.npy", all_rows_npy)
numpy.save("/home/blin/PycharmProjects/nlb/train_split_valid1.npy", all_rows_valid_npy)
print("yes", yes)
print("no", no)
#print(all_rows_npy[49838])
