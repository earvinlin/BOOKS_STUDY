import sys

if len(sys.argv) < 2 :
    print("You need input one parameter(fmt : theFileName)")
    print("syntax : C:\\python3 replaceString.py stockslist.txt")
    sys.exit()

theInputFileName = sys.argv[1]
# 讀取檔案內容
with open(theInputFileName, 'r', encoding='utf-8') as f:
    content = f.read()

# 取代指定字元
new_content = content.replace('></', '>\n</')

# 寫回檔案（覆蓋原始內容）
with open('example.txt', 'w', encoding='utf-8') as f:
    f.write(new_content)


with open('example.txt', 'r', encoding='utf-8') as f:
    content = f.read()

new_content = content.replace('</option><option value="', '').replace('">', '') \
.replace('<datalist id="dlSTOCK_ID_NM<option value="', '') \
.replace('</option>', '').replace('</datalist>', '')
with open('example-1.txt', 'w', encoding='utf-8') as f:
    f.write(new_content)