import numpy as np

class test(object):
    def __init__(self,id):
        self.id=id

array=[test(id) for id in range(10)]
for data in array:
    print(data.id)