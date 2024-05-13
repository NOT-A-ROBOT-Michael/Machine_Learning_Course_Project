import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
import os
import pathlib
import run_training

def main():
    run_training.train_models()
    
    for file in os.listdir(find_file_path()):
        if file.startswith("web_app"):
            exFile = find_file_path() + "\\web_app.py"
            exec(open(exFile).read())
            break

    
def find_file_path():
    return str(pathlib.Path(__file__).parent.resolve()) + "\\src_files"   


main()