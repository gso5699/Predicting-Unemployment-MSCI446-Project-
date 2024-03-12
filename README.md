# Recursive Neural Network
The code for the RNN was adapted from https://encord.com/blog/time-series-predictions-with-recurrent-neural-networks/#:~:text=Recurrent%20Neural%20Networks%20(RNNs)%20bring,genuine%20potential%20of%20predictive%20analytics.

## Setup
1) Create a new environment with the following mandatory packages:
    ```html
    numpy
    opencv
    tensorboard
    torch
    torchmetrics
    torchvision
    tqdm
    typer
    matplotlib
    ```
2) Activate corresponding virtual enviornment
3) Navigate to repository directory

## How To Run
- To run, 
    Either type the following in the Console
    ```html
    python run.py
    ```
    or 
    Directly run the run.py file. 
- The run.py file will call the ```run_rnn``` file situated in ```run_rnn.py```
- The trained model will output in a folder with the following format:
    ```html
    outputs/<model>/<time_stamp>

    Example: outputs/2023-03-11_093220
    ```
    The output folder will contain the following subfolders:
    - ```tensorboard```: Contains data for visualization in Tensorboard.
    - ```best```: Contains the file ```best_model.pth```, the best model generated during the training run
    - ```model_info.txt```: Text file with describing the hyperparameters and errors for best_model

    If the the training was run with ```--enable-checkpoints``` the output folder will also contain the following subfolder:
    - ```checkpoint```: Contains checkpoint files. These checkpoints files are generated every 100 epochs during training

### Deploying a model on test dataset
To deploy a model on the test dataset:
1) Open the Jupyter notebook ```deploy_model.ipynb``` 
2) In the first code block, paste the path to the model.
    Example: model_path = 'outputs/2024-03-11_015721/best/best_model.pth'