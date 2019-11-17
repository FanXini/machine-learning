import numpy as np
import cv2

screenshot = 'Pictures1/12.png'
img = cv2.imread(screenshot)
height,width=img.shape[0:2]
res = cv2.resize(img,(width//2, int(height//2.5)), interpolation = cv2.INTER_CUBIC)
image_np = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
#cv2.imshow("image",res)
cv2.imshow('image',image_np)
k = cv2.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()
elif k == ord('s'): # wait for 's' key to save and exit
    cv2.imwrite('messigray.png',img)
    cv2.destroyAllWindows()