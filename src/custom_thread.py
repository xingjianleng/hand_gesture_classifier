from threading import Thread
from typing import Callable


class CustomThread(Thread):
    def __init__(self, f: Callable, *args):
        Thread.__init__(self)
        self.func = f
        self.rtn = None
        self.args = args

    def run(self):
        rtn_value = self.func(*self.args)
        # store data in an instance variable
        self.rtn = rtn_value
