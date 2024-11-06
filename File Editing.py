# بسم الله الرحمن الرحيم

def write():
    obj1 = open("file.txt", "w")  # "w" mode allows you to write text in a NEW file, if file name exists; OVERWRITES it
    obj1.write("Can you see this? ")
    obj1.write("Are you working? ")
    obj1.write(f"100 200 300 3.14\nhello\nbye")
    obj1.close()


write()


def read_me():
    obj2 = open("file.txt", "r")  # "r" mode allows you to access and read an already created text file
    text1 = obj2.read(11)  # Reads the first 11 characters in the file "file.txt"
    text2 = obj2.readline()  # Reads the remaining text on the same line where "obj2.read(11)" ended at. (Reads text after the 11th character till the end of the line)
    text3 = obj2.readlines()  # Reads all the line(s) *under* the line where "obj2.read(11)" ended at, and packages them all into a list
    print(f"text1 = {text1}\ntext2 = {text2}text3 = {text3}\n")
    obj2.close()


read_me()


def add_me():
    obj3 = open("file.txt", "a")  # "a" mode allows you to add/append new text at the END of the file (end of last line in file) (if file name doesn't exist, it will create a new file)
    obj3.write("New Information!")  # Then you can add/write text in that new file normally. (I feel like "a" is much better than "w"; both allows you to add text,
    obj3.close()  # however; "w" will overwrite the file if it exists, whereas "a" will only add new text to it)


add_me()
read_me()
