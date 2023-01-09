#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

import requests
import boto3
from boto3sts import credentials as creds
import urllib.parse


session = creds.assumed_session("infncloud", endpoint="https://minio.cloud.infn.it/",
                                verify=True)
s3 = session.client('s3', endpoint_url="https://minio.cloud.infn.it/", config=boto3.session.Config(signature_version='s3v4'), verify=True)

#### assumiamo di voler scrivere il file stageout-file-1.root 
# in test dentro al bucket cygno-analysis. (notare anche in questo 
# caso la durata attesa di validità)
url_out = s3.generate_presigned_post(
                               'cygno-analysis',
                               'test/test.txt',
                               ExpiresIn=3600)

print(url_out)