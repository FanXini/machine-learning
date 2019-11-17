import cv2

path="screenshot.png"
img=cv2.imread(path)
img_object=img[921:1132,759:835]
cv2.imshow('image',img_object)
k = cv2.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()
elif k == ord('s'): # wait for 's' key to save and exit
    cv2.imwrite('messigray.png',img)
    cv2.destroyAllWindows()