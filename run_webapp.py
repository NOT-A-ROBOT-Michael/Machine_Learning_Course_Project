import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

from src_files import web_app
import runtraining

def main():
    print("///////////////start///////////////")
    runtraining.train_models()
    web_app
    print("///////////////stop///////////////")
    
    
    
def find_file_path():
    return str(pathlib.Path(__file__).parent.resolve())   

main()