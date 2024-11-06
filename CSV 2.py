import csv

with open("CSV.txt") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=" ")  # "delimiter" is the thing that will space out the words in a text (the stopper) (if deli="," --> everytime it sees a "," it will combine the previous words together, and then start over)
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print(f"Column names are: {' | '.join(row)}")  # Adds elements in "row" and separates them with commas (.join() adds elements in a list together)
            line_count += 1
        else:
            print(f"{row[0]} works in the {row[1]} department, and was born in {row[2]}")
            line_count += 1
            print("Processed", line_count, "lines.")
