import sys
import os
import time
from model import load_model

def main():
    print("Testing load_model caching...")

    # First call - should load from disk
    start = time.time()
    model1 = load_model("./model.pkl")
    end = time.time()
    print(f"First load time: {end - start:.4f}s")

    # Second call - should be cached
    start = time.time()
    model2 = load_model("./model.pkl")
    end = time.time()
    print(f"Second load time: {end - start:.4f}s")

    if model1 is model2:
        print("PASS: Objects are identical (cached)")
    else:
        print("FAIL: Objects are different (reloaded)")
        sys.exit(1)

if __name__ == "__main__":
    main()
