## Source code
The source code and the logs (inference_result) are organized as follows:

    ├──
        ├── ./data/  # Contains the keyword lists and preprocessing functions to create the JSONs combining file_name and corresponding GT captions
        ├── ./language_model/ # Stores the code of the language model
        ├── ./clip/  # Contains the code of the audio CLIP models. Their checkpoints can also be stored here
        ├── ./evaluation/ # Stores the code to create the plots and tables of the dissertation
        ├── ./sh_folder/ # Contains the shell-scripts to infer and to create the tables / plots
        ├── ./inference_result/ # Contains the inferenced results (one JSON, one CSV and one HTML per run) each run is uniquely identified by the timestamp
            ├── /facebook/opt-1.3b
                ├── /AudioCaps/excludes_prompt_MAGIC/
                    ├── evaluation 
                    ├── output_jsons
                    ├── output_tables
                ├── /clotho_v2.1/excludes_prompt_MAGIC/
                    ├── evaluation 
                    ├── output_jsons
                    ├── output_tables
        ├── ./softlinks/ # Store the softlinks to or the audio files themselves here!!!
        ├── inference_magic.py # The central python script combining the data, the models and that stores the results
        └── sound_obj_generator.py # This program created the ChatGPT keyword list
     

