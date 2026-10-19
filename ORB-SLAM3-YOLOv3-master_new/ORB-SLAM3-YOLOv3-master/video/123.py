import os
 
def getName(num):
    if num <10:
        strRes = '0000000' + str(num)
    elif num <100:
        strRes = '000000' + str(num)
    elif num <1000:
        strRes = '00000' + str(num)
    elif num < 10000:
        strRes = '0000' + str(num)
    return strRes
 
file_object = open('outdoor.txt','w')
Ostr = ''
num = len(os.listdir('test/Outdoor/data'))
for i in range(1,num+1):
    name = getName(i)
    Ostr = name + "\n"
    file_object.writelines(Ostr)
file_object.close()
print('file compelete')

