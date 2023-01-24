#!/usr/bin/env python3.6
import os

for name, value in os.environ.items():
    print("{0}: {1}".format(name, value))

