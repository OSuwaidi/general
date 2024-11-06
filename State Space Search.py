# بسم الله الرحمن الرحيم

from Shuffler import shuffler

state = ['H', 'H', 'H', 'H', 'H', 'T', 'T']


def search_space(state):
    unique_configurations = {tuple(shuffler(state, sr=1)) for _ in range(1000)}
    return len(unique_configurations)


print(f'Possible configurations/states: {search_space(state)}')
