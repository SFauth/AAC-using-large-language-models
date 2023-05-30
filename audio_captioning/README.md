## Source code
In this directory, we provide an overview over each folder of the source code. 

    ├──
        ├── ./data/  # Contains the keyword lists and preprocessing functions to create the JSONs combining file_name and corresponding GT captions
        ├── ./language_model/ # Stores the code of the language model
        ├── ./clip/  # Contains the code of the audio CLIP models. Their checkpoints can also be stored here
        ├── ./evaluation/ # Stores the code to create the plots and tables of the dissertation
        ├── ./sh_folder/ # Contains the shell-scripts to infer and to create the tables / plots
        ├── ./inference_result/ # Contains the inferenced results (one JSON, one CSV and one HTML per run) each run is uniquely identified by the timestamp
        └── sound_obj_generator.py # This program created the ChatGPT keyword list
     

