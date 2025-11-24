# Kaggle Playground S4E2 - Obesity Risk Prediction

## Setup
1.  **Data**: Download `train.csv` and `test.csv` from [Kaggle](https://www.kaggle.com/competitions/playground-series-s4e2/data) and place them in the `data/` folder.
2.  **Environment**: Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Running the Code
To train the model and generate a submission:
```bash
cd src
python train.py
```

## Project Structure
-   `data/`: Place dataset files here.
-   `src/`: Source code for preprocessing and modeling.
-   `notebooks/`: Jupyter notebooks for EDA.
-   `submission.csv`: Generated submission file.
