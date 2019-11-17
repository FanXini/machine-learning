import os

def pull_screenshot(path):
    num=8
    os.system("adb shell /system/bin/screencap -p /sdcard/screenshot8.png")
    os.system("adb pull /sdcard/screenshot"+str(num)+".png {0}".format(path))

pull_screenshot("Png/haha.png")
