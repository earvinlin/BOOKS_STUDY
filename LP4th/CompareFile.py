import re
import sys

print(len(sys.argv))

if len(sys.argv) < 4 :
    print("You need input two parameter(fmt : inputfile1 inputfile2 outfile3)")
    print("syntax : C:\python CompareFile.py 22Q1_stock_db.txt 22Q1_stock_files.txt result.txt ")
    sys.exit()

try :
    list1 = []
    list2 = []

    input_file1 = sys.argv[1]
    input_file2 = sys.argv[2] 
    output_file = sys.argv[3]
    print("=== Start the Programs ===")
   
    infile1 = open(input_file1, 'r')  
    for line in infile1.readlines():
        list1.append(line)
    infile1.close()

    infile2 = open(input_file2, 'r')  
    for line in infile2.readlines():
        list2.append(line)
    infile2.close()

    both = [x for x in list1 if x in list2]	# 兩個list都存在
    diff = [y for y in (list1 + list2) if y not in both] # 兩個list中不同的元素 

    outfile = open(output_file, 'w')
    for item in diff:
        outfile.write("%s" % item)

    print('資料儲存完成!!')
    outfile.close()


except IOError as err :
    print('Fie error : ' + str(err))
