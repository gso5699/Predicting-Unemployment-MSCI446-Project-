from datetime import datetime
from pathlib import Path
import typer

from run_rnn import *

app = typer.Typer()
@app.command()

def run(
    save_dir: Path = Path('outputs/'), 
    enable_checkpoints: bool = True,
    model_type: str = "RNN"
):
    """
    Runs a training process

    Parameters:
    - save_dir (Path): Directory where the model and Tensorboard files will be saved. 
        Default is 'outputs/'.
    - enable_checkpoints (bool): Whether to save model checkpoints during training. 
        Default is True, meaning checkpoints are enabled.
    - model_type (str): A string indicating the type of model.
        Default is 'RNN'

    """
    # SAVE_PATH and Tensorboard setup
    SAVE_PATH = Path(f'{save_dir}/{datetime.now().strftime("%Y-%m-%d_%H%M%S")}')
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir=f'{SAVE_PATH}/tensorboard')
    
    # Run RNN
    if model_type == 'RNN':
        run_rnn(SAVE_PATH, writer, enable_checkpoints)

if __name__ == '__main__':
    app()