
import re
from datetime import datetime

def strm():
# initializing string
    

    # printing original string
    print("The original string is : " + str(test_str))

    # searching string
    match_str = re.search(r'\d{4}-\d{2}-\d{2}', test_str)

    # computed date
    # feeding format
    res = datetime.strptime(match_str.group(), '%Y-%m-%d').date()
    return res

test_str = "gfg at 2021-01-04"

class test:
    b=0
    def __init__(self,x):
        self.x = x
        def bl(x):
            return x*x
        self.x=bl(self.x)
        # test.b=test.b+self.x
    def show(self):
        o=self.b
        def so(o):
          print(o)
    def s(self):
        print(self.x)

class no(test):
    def seen(self):
        print(self.b)
