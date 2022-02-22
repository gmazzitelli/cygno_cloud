#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

import requests
import boto3
from boto3sts import credentials as creds
import urllib.parse

#### assumiamo di voler scrivere il file stageout-file-1.root 
# in test dentro al bucket cygno-analysis. (notare anche in questo 
# caso la durata attesa di validità)
url_out = s3.generate_presigned_post(
                               'cygno-analysis',
                               'test/test.txt',
                               ExpiresIn=3600)

print(url_out)