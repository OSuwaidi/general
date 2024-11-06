# بسم الله الرحمن الرحيم

def rem_none(tup):
    tup = list(tup)
    for n in range(len(tup)):
        if n % 2 == 0:
            temp = list(tup[n])
            for i in range(len(temp)):
                if temp[i] is None:
                    temp[i] = ''
            tup[n] = tuple(temp)
    tup = tuple(tup)
    return tup


data = (('Robert', 'Hoit', None, None, 'TX'), {'fname': 0, 'lname': 1, 'Age': 2, 'Gender': 3, 'State': 4},
        ('James', 'Burns', 34, 'M', 'CA'), {'fname': 0, 'lname': 1, 'Age': 2, 'Gender': 3, 'State': 4},
        ('Matt', 'Dan', 45, None, 'NY'), {'fname': 0, 'lname': 1, 'Age': 2, 'Gender': 3, 'State': 4})
print(rem_none(data))
