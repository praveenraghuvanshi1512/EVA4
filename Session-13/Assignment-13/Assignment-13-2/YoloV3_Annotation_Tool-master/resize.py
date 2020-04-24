import os
import cv2
import sys

# global vars to inform user whether any of his pictures were not processed well
# don't thank me, I've been through hardship of annoting whole bunch of data then augment to find myself
# with rotten missing dataset...

sucess = 0
fail = 0

def resize(path_to_file):
    scaleFactor = 0.99
    
    if not os.path.exists(path_to_file):
        print('File not found')
        fail += 1
        return
    img = cv2.imread(path_to_file)

    filename = path_to_file.split('/')[-1]
    if filename.split('.')[-1] != 'jpg':
        print('wrong extension for file, please change')
        fail += 1
        return

    if not os.path.exists('Images/tmp'):
        os.makedirs('Images/tmp')

    # 1300 and 800 are arbitrary size that fits in a full screen gui
    x,y = len(img[1]),len(img[0])
    while x > 1300 or y > 800: x,y = x*scaleFactor,y*scaleFactor
    resized = cv2.resize(img,(x,y),interpolation = cv2.INTER_AREA)
    cv2.imwrite('Images/tmp'+filename,resized)
    sucess += 1

def main():
    
    array = sys.argv[1:]
    if len(array) == 0:
        print('Not the right number of arguments, use -h for usage recommendations')
        return
    elif array[0] == '-h':
        if len(array) > 1: print('Not the right number of arguments, for this feature, only -h args is accepted')
        else: print('-h: for helper ; -r path_to_file: for resize of a specific file ; -r path_to_file *: for resize of all files in the folder of the path_to_file given')
    elif array[0] == '-r':
        if len(array) > 3: print('Not the right number of arguments, use -h for usage recommendations')
        elif not os.path.exists(array[1]) : print('Please enter the path to file, use -h for usage recommandations')
        elif len(array) == 3:
            if array[2] == '*':
                folder = array[1].split('/')[:-1]
                path = ''
                for string in folder: path = path + '/' + string
                path = path[1:]
                for f in os.listdir(path): resize(path+'/'+f)
                print('Among all files there are, '+str(sucess)+' success and '+str(fail)+' failure')
            elif: print('Not correct usage, use -h for usage recommendation' )
        elif len(array) == 2:
            if not os.path.exists(array[1]) : print('Please enter the path to file, use -h for usage recommandations')
            else: resize(array[1])
        else: print('Not the right number of arguments, use -h for usage recommendations')
    else: print('Not correct usage, use -h for usage recommendation' )
if __name__ == '__main__':
    main()
