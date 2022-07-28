from pathlib import Path


def verify_csv(csv_path):
    path = Path(csv_path)
    for file_name in path.rglob("*.csv"):
        with open(file_name, "r") as fr:
            for i, _ in enumerate(fr):
                pass
            if i != 374:
                print(file_name, i)


def verify_txt(txt_path):
    path = Path(txt_path)
    for file_name in path.rglob("*.txt"):
        with open(file_name, "r") as fr:
            for i, _ in enumerate(fr):
                pass
            if i != 9334:
                print(file_name, i)


if __name__ == "__main__":
    check_path = "../data"
    verify_txt(check_path)
