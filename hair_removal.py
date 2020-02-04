import cv2

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
cv2.imshow("blackhat", blackhat)

# Create contours & inpaint
ret, thresh = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
inpaint = cv2.inpaint(image, thresh, 1, cv2.INPAINT_TELEA)
cv2.imshow("inpaint", inpaint)

cv2.waitKey(0)
cv2.destroyAllWindows()