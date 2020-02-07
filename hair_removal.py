import cv2
import os

path = r'C:\Users\ravee\Jupyter\Skin-Cancer-Classification\images\melanoma\ISIC_0024323.jpg'
# path = r'C:\Users\ravee\Jupyter\Skin-Cancer-Classification\images\bcc\ISIC_0024431.jpg'
image = cv2.imread(path)
cv2.imshow("raw", image)

# Convert
grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# Create kernel & perform blackhat filtering
kernel = cv2.getStructuringElement(1,(17,17))
print(kernel)
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
blackhat = cv2.morphologyEx(grayscale, cv2.MORPH_BLACKHAT, kernel)
# blackhat = cv2.morphologyEx(grayscale, cv2.MORPH_TOPHAT, kernel)
# cv2.imshow("blackhat", blackhat)

# Create contours & inpaint
ret, thresh = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
inpaint = cv2.inpaint(image, thresh, 1, cv2.INPAINT_TELEA)
cv2.imshow("inpaint", inpaint)

# Contouring
hsv = cv2.cvtColor(inpaint, cv2.COLOR_BGR2HSV)
gray = cv2.cvtColor(inpaint, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (17, 17), 32)
ret, thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnt = max(contours, key=cv2.contourArea)
# Draw Contours
x,y,w,h = cv2.boundingRect(cnt)
cv2.rectangle(inpaint,(x,y),(x+w,y+h),(0,255,0),2)
cv2.drawContours(inpaint, cnt, -1, (0, 0, 255), 2)
cv2.putText(inpaint, 'skin_lesion', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (36,255,12), 2)
# Image View
# cv2.imshow('raw', raw)
# cv2.imshow('blur', blur)
# cv2.imshow('frame',img)
# cv2.imshow('thresh',thresh
savepath = r'C:\Users\ravee\Jupyter\Skin-Cancer-Classification\images\thresh\melanoma'
filename = "hair_removal.jpg"
cv2.imwrite(os.path.join(savepath, filename), thresh)

cv2.waitKey(0)
cv2.destroyAllWindows()