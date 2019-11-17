import numpy as np
import cv2
path="screenshot.png"
img=cv2.imread(path)
H,W=img.shape[0:2]
for scale in np.linspace(1-0.3, 1+0.3, 10)[::-1]:
    resized=cv2.resize(img,(int(W*scale),int(H*scale)))
    cv2.imshow('image', resized)
    k = cv2.waitKey(0)
    if k == 27:  # wait for ESC key to exit
        cv2.destroyAllWindows()
    elif k == ord('s'):  # wait for 's' key to save and exit
        cv2.imwrite('messigray.png', img)
        cv2.destroyAllWindows()