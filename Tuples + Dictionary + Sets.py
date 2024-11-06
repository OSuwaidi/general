# بسم الله الرحمن الرحيم

d1 = {'apples': 3, 'oranges': 2, 'milk': 5}
print(d1.items())  # Gives back the dictionary as a tuple in a list (for tuple unpacking)
print(d1)
d1_reversed = {value: key for key, value in d1.items()}  # Used tuple unpacking!!! (for a, b in zip(L1, L2):)
print(d1_reversed, "\n")  # Switched/reversed keys with values and values with keys

T1 = (1, 2, 'cat', 'dog', 'cat')
T2 = 2, 4, 8
T3 = T1 + T2
print(T3)
print(type(T3))
print(len(T3))
print(max(T2))  # Has to be numbers only
print(sum(T2))
print(T1.index('dog'))
print(T3.count(2))
print(T1[4])
print(T2[-1])
print(T3[::-1])
print(T1 * 2)
T11 = list(T1)
print(type(T11), "\n")


S1 = {1, 2, 3, 3, 3, 3}  # "Sets" only contain UNIQUE values
print(S1)
S1.add(55)
print(S1)
S2 = {3, 5, 7, 5}
print(S2.issuperset(S1))  # Would be True if "S2" contained all the elements in "S1"
print(S1.union(S2))  # Combines unique values from S1 and S2
print(S1.intersection(S2))  # Shows what's common in both sets. Order doesn't matter
print(S2.difference(S1))  # Present in S2, not S1
print(S1.difference(S2))  # Present in S1, not S2
print(S1.symmetric_difference(S2), "\n")  # Outputs all the unique elements in both (not shared)


grocery = {'apples': 5, 'oranges': 3, 'dates': 7}
print(grocery)
grocery['bananas'] = 1
print(grocery)
del grocery['dates']
print(grocery)
print(grocery.keys())
print(grocery.values())
print(grocery.items())
print(grocery.get('apples'))

for key in grocery:
    print(key, "=", grocery[key])

dic_in_dic = {"old dictionary": grocery}
print(dic_in_dic)
print(dic_in_dic["old dictionary"]['oranges'])
print(dic_in_dic.values())
