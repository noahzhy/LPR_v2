import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
## 根据每行和每列的黑色和白色像素数进行图片分割。

def cv2_imread(file_path):
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
    return cv_img

# 1、读取图像，并把图像转换为灰度图像并显示
img_ = cv2_imread(r'test\03너8165_1632988910000.jpg')
img = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)  # 转换了灰度化

# 图像阈值化操作——获得二值化图
# ret, img_thre = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY_INV)
img_thre = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C , cv2.THRESH_BINARY_INV, 3, 2)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))  # 形态学处理:定义矩形结构
closed = cv2.dilate(img_thre, kernel, iterations=1)  # 闭运算：迭代5次

height, width = closed.shape[:2]
# 储存每一列的黑色像素数
v = [0] * width
# 储存每一行的黑色像素数
z = [0] * height
hfg = [[0 for col in range(2)] for row in range(height)]
lfg = [[0 for col in range(2)] for row in range(width)]
box = [0,0,0,0]

######水平投影  #统计每一行的黑点数，行分割#######
a = 0
emptyImage1 = np.zeros((height, width, 3), np.uint8)
for y in range(0, height):
    for x in range(0, width):
        if closed[y, x] == 0:
            a = a + 1
        else:
            continue
    z[y] = a
    a = 0

# 绘制水平投影图
l = len(z)
for y in range(0, height):
    for x in range(0, z[y]):
        b = (255, 255, 255)
        emptyImage1[y, x] = b

#根据水平投影值选定行分割点
inline = 1
start = 0
j = 0
# print(height,width)
# print(z)
for i in range(0,height):
    # inline 为起始位置标识，0.95 * width可自行调节，为判断字符位置的条件
    if inline == 1 and z[i] < 0.5 * width:  #从空白区进入文字区
        start = i  #记录起始行分割点
        #print i
        inline = 0
    # i - start > 3字符分割长度不小于3，inline为分割终止位置标识，0.95 * width可自行调节，为判断字符位置的条件
    elif (i - start > 3) and z[i] >= 0.5 * width and inline == 0 :  #从文字区进入空白区
        inline = 1
        hfg[j][0] = start - 2  #保存行分割位置
        hfg[j][1] = i + 2
        j = j + 1
####################### 至此完成行的分割 #################

#####对每一行垂直投影、分割#####
a = 0
for p in range(0, j):
    # 垂直投影  #统计每一列的黑点数
    for x in range(0, width):
        for y in range(hfg[p][0], hfg[p][1]):
            cp1 = closed[y,x]
            if cp1 == 0:
                a = a + 1
            else :
                continue
        v[x] = a  #保存每一列像素值
        a = 0
    print(v)
    # 创建空白图片，绘制垂直投影图
    l = len(v)
    emptyImage = np.zeros((height, width, 3), np.uint8)
    for x in range(0, width):
        for y in range(0, v[x]):
            b = (255, 255, 255)
            emptyImage[y, x] = b
    #垂直分割点
    incol = 1
    start1 = 0
    j1 = 0
    z1 = hfg[p][0]
    z2 = hfg[p][1]
    word = []
    for i1 in range(0,width):
        if incol == 1 and v[i1] <= 34 :  #从空白区进入文字区
            start1 = i1  #记录起始列分割点
            incol = 0
        elif (i1 - start1 > 3) and v[i1] > 34 and incol == 0 :  #从文字区进入空白区
            incol = 1
            lfg[j1][0] = start1 - 2   #保存列分割位置
            lfg[j1][1] = i1 + 2
            l1 = start1 - 2
            l2 = i1 + 2
            j1 = j1 + 1
            cv2.rectangle(img_, (l1, z1), (l2, z2), (255,0,0), 2)
cv2.imshow('original_img', img_)
cv2.imshow('erode', closed)
# cv2.imshow('chuizhi', emptyImage)
# cv2.imshow('shuipin', emptyImage1)
cv2.waitKey(0)
cv2.destroyAllWindows()
