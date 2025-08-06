import os
import sys

# add python path of Smart-Surveillance-System to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 3)))
sys.path.insert(0, parent_path)

def main():
    print(parent_path)

if __name__ == "__main__":
    main()