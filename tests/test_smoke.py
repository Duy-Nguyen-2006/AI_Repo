import subprocess, os, sys

def test_smoke_predict_import():
    # ensure modules import
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    import predict
    assert hasattr(predict, "predict_match")

if __name__ == "__main__":
    test_smoke_predict_import()
    print("smoke ok")


