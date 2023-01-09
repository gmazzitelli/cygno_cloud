#! /usr/bin/env python3

import boto3
import requests
from boto3sts import credentials as creds
import urllib.parse

def  main(fun, key, url, bucket, session, verbose):

    session = creds.assumed_session(session, endpoint=url,verify=True)
    s3 = session.client('s3', endpoint_url=url, config=boto3.session.Config(signature_version='s3v4'), verify=True)
    if fun == "get":
        url_out = s3.generate_presigned_url('get_object', 
                                        Params={'Bucket': bucket,
                                                'Key': key}, 
                                        ExpiresIn=3600)
    elif fun == "put":
        url_out = s3.generate_presigned_post(bucket, key, ExpiresIn=3600)
    else:
        url_out = ''

    print("Destination", url_out)
    
    # with open('stageout-file-1.root', 'rb') as f:
    #    files = {'file': ('stageout-file-1.root', f)}
    #    http_response = requests.post(url_out['url'], data=url_out['fields'], files=files)


if __name__ == "__main__":
    from optparse import OptionParser
    parser = OptionParser(usage='usage: %prog\t [-ubsv] get/put Key')
    parser.add_option('-u','--url', dest='url', type='string', default='https://minio.cloud.infn.it/', 
                      help='url [https://minio.cloud.infn.it/];');
    parser.add_option('-b','--bucket', dest='bucket', type='string', default='cygno-sim', 
                      help='bucket [cygno-sim];');
    parser.add_option('-s','--session', dest='session', type='string', default='infncloud-iam', 
                      help='shot name [infncloud-iam];');
    parser.add_option('-v','--verbose', dest='verbose', action="store_true", default=False, help='verbose output;');
    (options, args) = parser.parse_args()
    
    if len(args) < 2:
        print(args, len(args))
        parser.error("incorrect number of arguments")

    else:
        main(args[0], args[1], options.url, options.bucket, options.session, options.verbose)
