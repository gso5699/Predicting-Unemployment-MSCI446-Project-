from datetime import datetime
from pathlib import Path
import typer

from run_rnn import *
from preprocess_data import *

app = typer.Typer()
@app.command()

def run(
# Include flags here 
    save_dir: Path = Path('outputs/'),
    enable_checkpoints: bool = False
):
    # SAVE_PATH and Tensorboard setup
    SAVE_PATH = Path(f'{save_dir}/{datetime.now().strftime("%Y-%m-%d_%H%M%S")}')
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir=f'{SAVE_PATH}/tensorboard')
    
    data_clean, data_raw, scaler = get_preprocessed_data()
    run_rnn(data_clean, data_raw, scaler, SAVE_PATH, writer, enable_checkpoints)

    # Set path_to_saved_model to the best_model generated from the training, if it exists
    model_path = Path(f'{SAVE_PATH}/best/best_model.pth')
    if model_path.exists():
        path_to_saved_model = model_path


if __name__ == '__main__':
    app()