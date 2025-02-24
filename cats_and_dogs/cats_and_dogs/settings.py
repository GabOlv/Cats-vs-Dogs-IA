import os 

def get_path_to_venv():
    # Construct the path to the .venv folder (join with the current directory up to the root of the project)
    venv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.venv'))
    return venv_path

def get_path_to_data():
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'img'))
    return data_path\
    
def get_path_to_root():
    root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    return root_path

def get_cats_path():
    # Construct the path to the Cat Folder file inside Data
    cat_path = os.path.join(get_path_to_data(), 'Cat')
    return cat_path

def get_dogs_path():
    # Construct the path to the Cat Folder file inside Data
    dog_path = os.path.join(get_path_to_data(), 'Dog')
    return dog_path