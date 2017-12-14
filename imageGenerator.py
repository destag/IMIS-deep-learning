import cv2
import numpy as np
import random


def noise(img):
    return img + np.random.randint(-8, 13, size=(50, 50))


ilosc_kopi = 250     #maksymalnie 999

for i in range(1, 38):
    im = cv2.imread('./alfabet_i_cyfry/{0}/{0}-001.png'.format(str(i).rjust(2, "0")), 0)
    for j in range(2, ilosc_kopi):
        rows, cols = im.shape
        M = cv2.getRotationMatrix2D((cols/2, rows/2), random.randint(-12, 12), 1)         #rotacja w zakresie -12, 12 st
        dst1 = cv2.warpAffine(im, M, (cols, rows), borderMode=cv2.BORDER_REPLICATE)

        M = np.float32([[1, 0, random.randint(-2, 2)], [0, 1, random.randint(-2, 2)]])    #translacja od -2 do 2 pixeli
        dst2 = cv2.warpAffine(dst1, M, (cols, rows), borderMode=cv2.BORDER_REPLICATE)

        if random.randint(0, 1):
            kernel = np.ones((random.randint(1, 3), random.randint(1, 3)), np.uint8)        #erozja
            dst3 = cv2.erode(dst2, kernel, iterations=1)
        else:
            kernel = np.ones((1, 1), np.uint8)                                               #dylacja
            dst3 = cv2.dilate(dst2, kernel, iterations=1)

        kernel = np.ones((random.randint(1, 2), random.randint(1, 2)), np.float32)          #blur
        dst4 = cv2.filter2D(dst3, -random.randint(1, 2), kernel)

        if random.randint(0, 4):
            pts1 = np.float32([[0, 45], [40, 15], [0, 20]])
            pts2 = np.float32([[0, random.randint(45, 47)], [random.randint(39, 41), 15], [0, random.randint(19, 21)]])
            M = cv2.getAffineTransform(pts1, pts2)
            dst4 = cv2.warpAffine(dst4, M, (cols, rows), borderMode=cv2.BORDER_REPLICATE)       #transformacja afimiczna

        if not random.randint(0, 4):
            dst4 = cv2.resize(dst4, None, fx=random.uniform(1.0, 1.10), fy=random.uniform(1.0, 1.10),
                              interpolation=cv2.INTER_LINEAR)
            dst4 = dst4[0:50, 0:50]                                                             #zoom

        dst5 = noise(dst4)          #szum

        result = dst5

        cv2.imwrite('./alfabet_i_cyfry/{0}/{0}-{1}.png'.format(str(i).rjust(2, "0"), str(j).rjust(3, "0")), result)
