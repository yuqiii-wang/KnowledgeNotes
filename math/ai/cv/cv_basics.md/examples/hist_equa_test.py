from PIL import Image
import numpy as np


img = Image.open("testImg.png")

gray_img = img.convert("L") # to grayscale
gray_img.show()

n_rows = gray_img.size[1]
n_cols = gray_img.size[0]
n_i = np.array(gray_img.histogram())
p_xi = n_i / (n_rows * n_cols)
L = len(n_i)

c_xi = np.zeros(L)
for i in range(L):
    c_xi[i] = np.sum(p_xi[:i])

gray_img = np.array(gray_img)
histEquaImgArr =  np.zeros((n_rows, n_cols))
for r in range(n_rows):
    for c in range(n_cols):
        histEquaImgArr[r,c] = c_xi[gray_img[r,c]]

histEquaImg = Image.fromarray(np.uint8(histEquaImgArr * (L-1)))
histEquaImg.show()