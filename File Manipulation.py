# بسم الله الرحمن الرحيم

import os
import shutil

# Note: to change the current working directory, use: "os.chdir('new_path')"
# Note: to know the current working directory, or to make sure it has changed after using "os.chdir()", use: "os.getcwd()"

source_path = "/Users/darke/OneDrive/Documents/Python Scripts/CV/cats/train"
target_path = "/Users/darke/OneDrive/Documents"

source_file = os.listdir(source_path)  # Returns a list containing all the files/folders stored in that source path (directory)
                                       # Note: "os.listdir()" will return items in whatever "working" directory the script was called from!

# To move files from one destination to another:
target = 'test.jpg'

for image in source_file:  # Loops over all the files/folders under "source_file"
    if image == target:
        shutil.move(f'{source_path}/{image}', f'{target_path}')
        break


# To delete files within a folder/directory:
for image in source_file:
    if image[0] == 'd':  # If image starts with the letter 'd', then delete it
        shutil.rmtree(f'{source_path}/{image}')

'''If we wanted to remove multiple files contained in a list/tuple at once: "map(shutil.rmtree, list_files)" '''

# To rename files:
'''os.rename(initial_name, target_name)'''

# To get the file's extension:
'''file_name, file_extension = os.path.splitext('path/to/file.ext')'''
