# Introduction
For our project, we created several models to predict unemployment. 

The main focus of our project was to implement basic regression models (i.e. Lasso Regression,Multiple Regression) to predict unemployment. These models used contemporaneous data, meaning they offered insights into how different economic factors influeced unemployment but lacked the capability to predict future unemployment due to the absence of data for the other economic factors at future timestamps.

As a secondary approach, we explored the implementation of a Recursive Neural Network (RNN). RNNs are capable of capturing sequential information and patterns over time. In an RNN, the input is a series of observations, and the target is a single response. While implementing an RNN model was not essential for achieving our initial project objectives, it provided additional depth to our learning.

---

# Regression Models for Predicting Unemployment
## Dataset Description

Below is a description of each feature:
| Feature                          | Description                                                               | Data Source                                                                                        |
|----------------------------------|---------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------|
| **Date**                         | The year and month of the observation.                                    | -                                                                                                 |
| **Unemployment**                 | Unemployment rate.                                                         | [Unemployment Data Source](https://www150.statcan.gc.ca/t1/tbl1/en/cv.action?pid=1410037401)     |
| **CPI_all-items**                | Consumer Price Index (CPI) for all items.                                 | [CPI All Items Data Source](https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=1810000601)   |
| **CPI_food**                     | Consumer Price Index for food items.                                       | [CPI Food Data Source](https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=1810000601)        |
| **CPI_shelter**                  | Consumer Price Index for shelter-related expenses.                         | [CPI Shelter Data Source](https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=1810000601)     |
| **CPI_household_op**             | Consumer Price Index for household operations.                             | [CPI Household Operations Data Source](https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=1810000601) |
| **CPI_clothing**                 | Consumer Price Index for clothing.                                         | [CPI Clothing Data Source](https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=1810000601)     |
| **CPI_transportation**           | Consumer Price Index for transportation.                                   | [CPI Transportation Data Source](https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=1810000601) |
| **CPI_health**                   | Consumer Price Index for health-related expenses.                          | [CPI Health Data Source](https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=1810000601)      |
| **CPI_rec**                      | Consumer Price Index for recreation, education, and reading.               | [CPI Recreation, Education, and Reading Data Source](https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=1810000601) |
| **CPI_alcohol**                  | CPI for alcoholic beverages, tobacco products, and recreational cannabis. | [CPI Alcohol Data Source](https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=1810000601)     |
| **CPI_no-food**                  | CPI excluding food items.                                                  | [CPI No Food Data Source](https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=1810000601)     |
| **CPI_no-food-and-energy**       | CPI excluding food and energy items.                                       | [CPI No Food and Energy Data Source](https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=1810000601) |
| **GDP**                          | Gross Domestic Product across all industries.                              | [GDP Data Source](https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=3610043401)             |
| **FEX_AUS** to **FEX_USA**       | Foreign exchange rates against various currencies.                         | [Foreign Exchange Rates Data Source](https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=3310016301) |
| **FEX_CAD**                      | Foreign exchange rate against the Canadian Dollar.                         | [Foreign Exchange Rate CAD Data Source](https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=3310016301) |
| **AWE_industrial-aggregate** to **AWE_public-admin** | Average Weekly Earnings (AWE) across different industries and sectors.  | [Average Weekly Earnings Data Source](https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=1410020301) |
| **IMT_import**                   | International Merchandise Trade Import value.                             | [Import Data Source](https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=1210001101)         |
| **IMT_export**                   | International Merchandise Trade, Export value.                             | [Export Data Source](https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=1210001101)         |
| **IMT_trade-bal**                | International Merchandise Trade, Trade balance.                            | [Trade Balance Data Source](https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=1210001101)  |
| **ITSI_total** to **ITSB_gov**   | International trade in services across various sectors (total, commercial services, travel, transport, and government). | [Imports Data Source](https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=1210014401), [Exports Data Source](https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=1210014401), [Balances Data Source](https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=1210014401) |
| **HPI_total** to **HPI_land**    | House Price Index (HPI) for total, house, and land prices.               | [House Price Index Data Source](https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=1810020501) |


These features provide a comprehensive view of the economic indicators and variables included in the dataset. 

## Description
TO BE WRITTEN; if the notebook file is already commented nicely, we don't have to include too much here. Just provide a brief outline on process, what we visualized, etc. 

---

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
    python RNN_Model/main.py
    ```
    or directly run the RNN_model/main.py file. 
- The main.py file will call the ```RNN_model/run_rnn``` function situated in ```RNN_model/run_rnn.py``` file
- The trained model will output in a folder with the following format:
    ```html
    RNN_model/outputs/<model>/<time_stamp>

    Example: RNN_model/outputs/2023-03-11_093220
    ```
    The output folder will contain the following subfolders:
    - ```tensorboard```: Contains data for visualization in Tensorboard.
    - ```best```: Contains the following:
        - ```best_model.pth```: the best model generated during the training run
        - ```model_info.txt```: Text file with describing the hyperparameters and errors for best_model

    If the the training was run with ```--enable-checkpoints``` the output folder will also contain the following subfolder:
    - ```checkpoint```: Contains checkpoint files. These checkpoints files are generated every 100 epochs during training

## Deploying a RNN model on test dataset
To deploy a model on the test dataset:
1) Open the Jupyter notebook ```RNN_model/deploy_rnn_model.ipynb``` 
2) In the first code block, paste the path to the model.
    Example: 
    ```html
    model_path = 'RNN_model/outputs/2024-03-11_015721/best/best_model.pth'
    ```
    

