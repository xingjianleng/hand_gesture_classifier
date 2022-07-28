from pathlib import Path


x = Path("./labelled_data")
for filename in x.rglob("*.csv"):
    count = 0
    with open(filename, "r") as f:
        for line in f:
            count += 1
        if count != 374:
            print(filename, count)
