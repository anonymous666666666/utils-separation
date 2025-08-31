# main.py
from utils import pipeline

def main():
    # Adjust these file names to whatever your input/output are
    pipeline("data.csv", "out/clean.csv")

if __name__ == "__main__":
    main()
