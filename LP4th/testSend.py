import os

def Gene():         #生成器函數
    print("ok")
    x = 100         
    print("in001: ", x)
    first = yield 50    #這裏就是send函數的關鍵
    # send所傳遞的值其實就是給 =號左邊的左值賦值
    print("in002: ", first)
    
    second = yield x  # 這裏試第二個斷點
    print("in003: ", second)
    
    z = 'third'
    third = yield z
    print("in004: ", third)
     

inst = Gene()               #創建生成器對象
os.system("pause")
output1 = inst.send(None)   #啟動生成器，運行到第一個yield
os.system("pause")
print("out001:", output1)   #這邊的output1獲得的是yield的返回值 50
os.system("pause")
output2 = inst.send(30)
os.system("pause")
print("out00２:", output2)
os.system("pause")
output3 = inst.send(None)
os.system("pause")