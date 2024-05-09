import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

import pathlib

from src_files import data_prep, train_first_model, feature_engineering, train_second_model 

def main():
    print("///////////////start///////////////")
    filepath = find_file_path()
    data_prep.dataprep(filepath)
    train_first_model.train_first_model(filepath)
    feature_engineering.feature_engineering(filepath)
    train_second_model.train_second_model(filepath)
    print("///////////////stop///////////////")
    
    
    
def find_file_path():
    return str(pathlib.Path(__file__).parent.resolve())   

main()
