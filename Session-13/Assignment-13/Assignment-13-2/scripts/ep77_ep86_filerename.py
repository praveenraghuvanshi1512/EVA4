# importing os module
import os


# Function to rename multiple files
def main():
    for count, filename in enumerate(os.listdir(".")):
        dst = "ep77_"+filename
        src = filename
        #dst = 'xyz' + dst

        # rename() function will
        # rename all the files
        os.rename(src, dst)

    # Driver Code


if __name__ == '__main__':
    # Calling main() function
    main()