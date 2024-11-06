# بسم الله الرحمن الرحيم

# ".isdigit()" checks whether the type of the variable is a digit or not (a word/character). If yes, return "True", else "False"
# Eg:
x = '918'
y = 'cat'
print(x.isdigit())
print(y.isdigit(), "\n")


# Create a Tic-Tac-Toe Game:
location = {1: " ", 2: " ", 3: " ", 4: " ", 5: " ", 6: " ", 7: " ", 8: " ", 9: " "}  # Define "location" as a global variable/dictionary to be used in multiple functions
win = 0  # Another global variable assignment


def display():
    global win  # IMPORTANT: Since we want the function "display()" to affect the value of "win" inside the function "game()", we had to global "win" UNDER "display()", so that changes on "win" WITHIN "display()" function actually change/affect "win" everywhere/globally
    print(f"  {location[1]}  |  {location[2]}  |  {location[3]}  \n{'-'*17}\n  {location[4]}  |  {location[5]}  |  {location[6]}  \n{'-'*17}\n  {location[7]}  |  {location[8]}  |  {location[9]}  \n")  # Or we could have done 3 print statements
    if location[1] == location[2] == location[3] != " " or location[4] == location[5] == location[6] != " " or location[7] == location[8] == location[9] != " " or location[1] == location[5] == location[9] != " " or location[3] == location[5] == location[7] != " " or location[1] == location[4] == location[7] != " " or location[2] == location[5] == location[8] != " " or location[3] == location[6] == location[9] != " ":
        win = 1  # IMPORTANT: If we didn't *global* "win" under "display()" function, calling "display()" UNDER "game()" would have no affect on the value of "win"  --> Since "display()" is UNDER "game()", then changing the value inside of "display()" (win = 1), would have no affect on the outside function "game()", because "display()" is ENCLOSED BY "game()", and (win = 1), is LOCAL assignment to "display()". Therefore, if you were to "print(win)" inside "display()", you would get win = 1, however if you "print(win)" inside "game()", right below "display()", it will print win = 0, because you only changed win = 1 LOCALLY inside of "display()", thus we made it global!!!
# Recall: # Enclosed local variables are variables that are not assigned locally within the function itself, but assigned within the enclosing function


def game():
    print("Lets Play Tic-Tac-Toe!\n")
    display()
    print("'X' starts first!")
    rnd = 2
    while win != 1:
        while rnd % 2 == 0:
            p1 = input('Choose location (1-9): ')
            while not p1.isdigit() or int(p1) not in range(1, 10):
                if not p1.isdigit():  # "not False = True"
                    print("Please enter a number")
                elif int(p1) not in range(1, 10):
                    print("Out of bounds!")
                p1 = input('Choose location (1-9): ')
            p1 = int(p1)
            if location[p1] == " ":
                location[p1] = "X"  # Here, we are changing the variable "location" LOCALLY, within "game()". (ENCLOSED local variable to "display()")
                rnd += 1
            else:
                print("Location occupied!")
        display()  # Since "display()" is ENCLOSED LOCALLY within "game()", the affect of the "location" assignment is shown, because the "location" assignment is an ENCLOSED LOCAL variable to "display()"
        if win == 1:
            print("'X' Player wins!")
            break
        if rnd >= 11:
            print("It's a Draw!")
            break
        while (rnd - 1) % 2 == 0:  # "(rnd - 1) % 2 == 0" ---> Used to detect odd numbers
            p2 = input('Choose location (1-9): ')
            while not p2.isdigit() or int(p2) not in range(1, 10):
                if not p2.isdigit():
                    print("Please enter a number")
                elif int(p2) not in range(1, 10):
                    print("Out of bounds!")
                p2 = input('Choose location (1-9): ')
            p2 = int(p2)
            if location[p2] == " ":
                location[p2] = "O"
                rnd += 1
            else:
                print("Location occupied!")
        display()
        if win == 1:
            print("'O' Player wins!")
            break


game()
