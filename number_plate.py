#packages
import numpy as np
import imutils
import cv2
import pytesseract
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

#Read,resize and show the image
image = cv2.imread('veh.jpeg')
image = imutils.resize(image, width=500)
cv2.imshow('Original image',image)


#pre-processing the image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Grayscale Conversion", gray)

gray = cv2.bilateralFilter(gray, 11, 17, 17)
cv2.imshow("Bilateral Filter", gray)#2

#detect the edges from the image
edged = cv2.Canny(gray, 170, 200)
cv2.imshow("Edge detection", edged)

cv2.waitKey(0)
cv2.destroyAllWindows()


(cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts=sorted(cnts, key = cv2.contourArea, reverse = True)[:30]
NumberPlateCnt = None

count = 0
for c in cnts:
        perimeter = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * perimeter, True)
        if len(approx) == 4:
            NumberPlateCnt = approx
            break

# Masking the part other than the number plate
mask = np.zeros(gray.shape,np.uint8)
new_image = cv2.drawContours(mask,[NumberPlateCnt],0,255,-1)
new_image = cv2.bitwise_and(image,image,mask=mask)
#cv2.imshow("lets see",new_image)#6
cv2.namedWindow("Final_image",cv2.WINDOW_NORMAL)
cv2.imshow("Final_image",new_image)
cv2.waitKey(0)
cv2.imwrite('new_image.png',new_image) #rewrite the image to the file
text=pytesseract.image_to_string(Image.open('new_image.png'))
print(text)
