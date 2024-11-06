# بسم الله الرحمن الرحيم

data = {'Thiem': (9, 2, 22, 104, 11, 106),
        'Medvedev': (10, 2, 11, 106, 10, 104),
        'Barty': (9, 2, 8, 74, 9, 76),
        'Osaka': (5, 2, 9, 74, 8, 74)}


def dic_sorted(dic):
    for r in range(len(dic) - 1):
        i = 0
        values = list(dic.values())
        while values[r][i] == values[r + 1][i]:
            i += 1
        if values[r][i] < values[r + 1][i]:
            key_shift(dic, values[r])

    return dic


def key_shift(dic, v1):
    keys = list(dic.keys())
    values = list(dic.values())
    temp_key = keys[values.index(v1)]
    del dic[temp_key]
    dic[temp_key] = v1


for i in range(5):  # Number of iterations depends on the complexity of your dictionary
    dic_sorted(data)
print(data)


'''Or simply:'''
print(sorted(data.items(), key=lambda x: x[1], reverse=True))