import csv

with open("test.txt") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=",")
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print(f"Column names are {','.join(row)}")  # Adds elements in "row" and separates them with commas (.join() adds elements in a list together)
            line_count += 1
        else:
            print(row[0], "works in the", row[1], "department, and was born in", row[2])
            line_count += 1
            print("Processed", line_count, "lines.")
