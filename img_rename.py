import os
 
# Function to rename multiple files
def main():
    i = 0
     
    for filename in os.listdir("driving_dataset2"):
        dst =str(i) + ".jpg"
        src ='driving_dataset2/'+ filename
        dst ='driving_dataset2/'+ dst
         
        # rename() function will
        # rename all the files
        os.rename(src, dst)
        i += 1
 
# Driver Code
if __name__ == '__main__':
     
    # Calling main() function
    main()
