import numpy as np
import cv2

def mathc_img(image,Target,value):
    img_rgb = cv2.imread(image)
    #img_rgb=cv2.Canny(img_rgb,20,80)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    #img_gray=cv2.Canny(img_rgb,20,80)
    template = cv2.imread(Target)
    template=cv2.cvtColor(template,cv2.COLOR_BGR2GRAY)
    h,w=template.shape[0:2]
    res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
    print(res.max())
    threshold = value
    loc = np.where( res >= threshold)#里面是预测精确度大于<0.9的起点坐标
    print(loc)
    print(zip(*loc[::-1]))
    for pt in zip(*loc[::-1]):
        print(pt)
        cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (7,249,151), 2)
        break
    img_rgb=cv2.resize(img_rgb,(int(img_rgb.shape[1]*0.4),int(img_rgb.shape[0]*0.4)))
    cv2.imshow('Detected',img_rgb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__=="__main__":
    num=1
    imagePath = "gogo.png"
    targetPath = 'target1/0.png'
    value = 0.9
    mathc_img(imagePath,targetPath,value)

