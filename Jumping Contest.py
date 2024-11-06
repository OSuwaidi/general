import matplotlib.pyplot as plt
import random

filename = input('Which file do you want to collect the competitors from?: ')  # "jump.txt"


class Player:
    win_length = {}
    def __init__(self, name, m, s, fp):
        self.name = name
        self.m = m
        self.s = s
        self.fp = fp
        self.best = 0
        self.wins = 0

    def __str__(self):
        return self.name + ' ' + str(self.wins) + ' ' + 'Wins'

    def jump(self):
        result = [0]
        foul = [1] * int(10 * self.fp) + [0] * int(10 * (1 - self.fp))
        for i in range(0, 6):
            if random.choice(foul) == 0:
                jump = 3 * (random.normalvariate(self.m, self.s))
                result.append(jump)
        self.best = max(result)

        return self.best


class Arena:
    players = []  # Can also do: "def __init__(self):" --> "self.players = []"

    def main(self):  # Will execute the 4 methods below automatically when you call "a.main()"
        print("Welcome to the competition, please enter how many simulations you would like to run!")
        self.readplayersfromfile()
        self.contest()
        self.sortStats()
        self.stats()

    def readplayersfromfile(self):
        infil = open(filename, 'r')
        line = infil.readline()
        while line != '':
            line = line.rstrip('\n')  # Removes "\n" if any
            parts = line.split()  # Splits sentences into a list of strings whenever it sees a space " "
            name = parts[0]
            Player.win_length[name] = []
            m = float(parts[2])
            s = float(parts[3])
            fp = float(parts[4])
            tmp = Player(name, m, s, fp)
            Arena.players.append(tmp)
            line = infil.readline()

        return Arena.players

    def contest(self):
        while True:
            try:
                x = int(input("How many competitions? "))
                break
            except ValueError:
                print("Only numbers please!")
        for rounds in range(0, x):
            for player in Arena.players:
                player.jump()
            bestjump = max(Arena.players, key=lambda jump: jump.best)
            Player.win_length[bestjump.name].append(bestjump.best)  # Add the best jump to list of winning player
            bestjump.wins += 1

        jump_list = []  # Contains jumps of all runs/simulations (for histogram)
        for lis in Player.win_length.values():
            jump_list += lis

        top_list = sorted(jump_list, reverse=True)[:10]
        print(f'\nTop 10 Jumps:')
        for i in top_list:
            for k in Player.win_length:
                if i in Player.win_length[k]:
                    print(f"{k}: {i:.4}")

        plt.hist(jump_list, color='y', ec='black')
        plt.title('Distribution of Winning lengths')
        plt.xlabel('Jump lengths')
        plt.show()

    def sortStats(self):
        return Arena.players.sort(key=lambda player: player.wins, reverse=True)

    def stats(self):
        print(f'\nResults:')
        for e in range(len(Arena.players)):
            print(Arena.players[e])  # Printing the objects/instances of the class themselves


a = Arena()
a.main()
