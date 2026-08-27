from pathlib import Path




def prepare_model(dataset_directory: str,
                  model_name: str):
    
    from .build import main

    main.build_model(Path(dataset_directory),
                     model_name)