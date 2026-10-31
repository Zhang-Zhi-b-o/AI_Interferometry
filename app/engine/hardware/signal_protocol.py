"""Signed decimal readings delimited by Enter, shared with the standalone collector."""
from decimal import Decimal
import re

SCAN = dict(zip(range(2, 12), "1234567890"))
SCAN.update({12: "-", 52: ".", 74: "-", 78: "+", 83: ".",
             71: "7", 72: "8", 73: "9", 75: "4", 76: "5", 77: "6",
             79: "1", 80: "2", 81: "3", 82: "0"})


class Frames:
    def __init__(self):
        self.aligned = False
        self.buffer = ""
        self.bad = False

    def feed(self, code, state):
        if state & 1:
            return None
        if code == 28:
            value = None
            if self.aligned and not self.bad and re.fullmatch(r"[+-]?\d+(?:\.\d+)?", self.buffer):
                value = Decimal(self.buffer)
            self.aligned, self.buffer, self.bad = True, "", False
            return value
        if code in SCAN:
            self.buffer += SCAN[code]
            if len(self.buffer) > 32:
                self.bad = True
        else:
            self.bad = True
        return None
