# Feature Description

Below is a description of each feature:

1. **Date**: The year and month of the observation.

2. **Unemployment_interpolated**: Interpolated values for unemployment rate. (Not currently used)

3. **Bank Interest Rate**: Interest rate set by the central bank or monetary authority.

4. **Change_in_GDP**: The change in Gross Domestic Product (GDP) compared to the previous period.

5. **Unemployment**: Unemployment rate.

6. **CPI_all-items**: Consumer Price Index (CPI) for all items.

7. **CPI_food**: Consumer Price Index for food items.

8. **CPI_shelter**: Consumer Price Index for shelter-related expenses.

9. **CPI_household_op**: Consumer Price Index for household operations.

10. **CPI_clothing**: Consumer Price Index for clothing.

11. **CPI_transportation**: Consumer Price Index for transportation.

12. **CPI_health**: Consumer Price Index for health-related expenses.

13. **CPI_rec**: Consumer Price Index for recreation, education, and reading.

14. **CPI_alcohol**: CPI for alcoholic beverages, tobacco products, and recreational cannabis

15. **CPI_no-food**: CPI excluding food items.

16. **CPI_no-food-and-energy**: CPI excluding food and energy items.

17. **GDP**: Gross Domestic Product across all industries

18. **FEX_AUS** to **FEX_USA**: Foreign exchange rates against various currencies (e.g., Australian Dollar, Brazilian Real, Chinese Yuan, etc.).

19. **FEX_CAD**: Foreign exchange rate against the Canadian Dollar.

20. **AWE_industrial-aggregate** to **AWE_public-admin**: Average Weekly Earnings (AWE) across different industries and sectors.

21. **IMT_import**: International Merchandise Trade Import value.

22. **IMT_export**: International Merchandise Trade, Export value.

23. **IMT_trade-bal**: International Merchandise Trade, Trade balance.

24. **ITSI_total** to **ITSB_gov**: International trade in services across various sectors (total, commercial services, travel, transport, and government).

25. **HPI_total** to **HPI_land**: House Price Index (HPI) for total, house, and land prices.

These features provide a comprehensive view of the economic indicators and variables included in the dataset.  


# Lasso Regression Model for Predicting Unemployment
To be written...


# Recursive Neural Network Model for Predicting Unemployment
The code for the RNN was adapted from https://encord.com/blog/time-series-predictions-with-recurrent-neural-networks/#:~:text=Recurrent%20Neural%20Networks%20(RNNs)%20bring,genuine%20potential%20of%20predictive%20analytics.

## Dataset Description
The unemployment data used in this RNN was sourced from Statistics Canada (https://www150.statcan.gc.ca/t1/tbl1/en/cv.action?pid=1410037401). 
It encompasses the total unemployment rates for all population centers and rural areas, presented as a percentage (%) for each month from January 2011 to January 2024.

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
- To run, either type the following in the Console
    ```html
    python main.py
    ```
    or directly run the main.py file. 
- The main.py file will call the ```run_rnn``` function situated in ```run_rnn.py``` file
- The trained model will output in a folder with the following format:
    ```html
    outputs/<model>/<time_stamp>

    Example: outputs/2023-03-11_093220
    ```
    The output folder will contain the following subfolders:
    - ```tensorboard```: Contains data for visualization in Tensorboard.
    - ```best```: Contains the following:
        - ```best_model.pth```: the best model generated during the training run
        - ```model_info.txt```: Text file with describing the hyperparameters and errors for best_model

    If the the training was run with ```--enable-checkpoints``` the output folder will also contain the following subfolder:
    - ```checkpoint```: Contains checkpoint files. These checkpoints files are generated every 100 epochs during training

## Deploying a model on test dataset
To deploy a model on the test dataset:
1) Open the Jupyter notebook ```deploy_model.ipynb``` 
2) In the first code block, paste the path to the model.
    Example: 
    ```html
    model_path = 'outputs/2024-03-11_015721/best/best_model.pth'
    ```
    

