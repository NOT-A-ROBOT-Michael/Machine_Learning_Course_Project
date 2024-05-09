import pathlib
from src_opt import feature_engineering, train_first_model, train_second_model, data_prep

def main():
    print(find_file_path())
    print("///////////////done/////////////")
    
    
    
def find_file_path():
    return str(pathlib.Path(__file__).parent.resolve())


def concat_to_file(directory =''):
    return find_file_path()+"/"+directory
    

main()
