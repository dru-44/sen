

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
