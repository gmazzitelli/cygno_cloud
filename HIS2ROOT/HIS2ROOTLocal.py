import os,sys

if not os.path.isfile('cygnus_lib.py'): 
    # to update library remove file cygnus_lib.py
    os.system('wget https://raw.githubusercontent.com/gmazzitelli/cygno_cloud/main/cygnus_lib.py')
sys.path.append('./')
import cygnus_lib as cy

def main(path):
    for file in os.listdir(path):
        if file.endswith(".HIS"):
            his_file = path+file
            print (his_file)
            filein, run = cy.ruttalo(his_file)
            print ("file {} done".format(filein))
    print("ALL DONE")


if __name__ == '__main__':
    import re
    sys.argv[0] = re.sub(r'(-script\.pyw|\.exe)?$', '', sys.argv[0])
    if len(sys.argv)==1:
        print ('Usage: <path>')
        sys.exit(1)
    else:
        sys.exit(main(sys.argv[1]))