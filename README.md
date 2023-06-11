[português (BR)](https://github.com/alyssonvidal/Hotel-Booking-Cancelations/blob/main/README_PT.md)

<center><img src="/images/hotel.jpg" alt="logo" width="600" height="480"/></center>

# Problem Statement

One of the common main problems the hotels facing are cancellations. There are several problems due to cancellations, including: operational problems, replanning of tasks, inaccuracy in revenue forecasts, poor optimization of resources such as occupancy of rooms..

This study contains informations about a Ciy Hotel and a Resort Hotel in the period between July 2015 and August 2017, the City Hotel is located in Lisbon, capital of Portugal and the Resort Hotel in Algarve, a coastal city also from Portugal.

[glossário](https://github.com/alyssonvidal/Hotel-Booking-Cancelations/blob/main/references/glossary_PT.md), [dataset](https://www.sciencedirect.com/science/article/pii/S2352340918315191)


# Objective

* Identify the main characteristics of guests who cancel bookings.
* Develop a Machine Learning Model capable of predicting which guests are most likely to cancel the booking.
* Present a possible business plan based on the results obtained.<br><br>

# Development Stages


[**Data Preprocessing**](https://github.com/alyssonvidal/Hotel-Booking-Cancelations/blob/main/notebooks/part01_preprocessing.ipynb)<br>
Loading Data, Duplicated Values, Missing Values, Strange Values, Data Types, Feature Engieneering.

[**Exploratory Data Analysis**](https://github.com/alyssonvidal/Hotel-Booking-Cancelations/blob/main/notebooks/part02_eda.ipynb)<br>
Mind Map Business, Hypotesis, Summary Table, Bivariated Analysis, Multivariated Analysis.

[**Data Preparation**](https://github.com/alyssonvidal/Bank-Marketing-Cluster/blob/main/notebooks/part03_model.ipynb)<br>
Encoding, Normalization, Feature Selection.

[**Modelo de Machine Learning**](https://github.com/alyssonvidal/Hotel-Booking-Cancelations/blob/main/notebooks/part04_model_lightgbm.ipynb)<br>
Models: LightGBM, XGBoost, Pytorch.

[**Feature Importance**](https://github.com/alyssonvidal/Bank-Marketing-Cluster/blob/main/notebooks/part03_model.ipynb)<br>
Optuna.

[**Feature Importance**](https://github.com/alyssonvidal/Hotel-Booking-Cancelations/blob/main/notebooks/part05_feature_importance.ipynb)<br>
Anova, Chi2, Embeeded Methods, SHAP

[**MLflow**](https://github.com/alyssonvidal/Hotel-Booking-Cancelations/blob/main/notebooks/part06_mlflow.ipynb)<br>
Some cool stuffs for practices....<br><br>

# Tools
Languages: Python, Shell Script<br>
IDE: Visual Studio Code, Jupyter Notebook.<br>
Frameworks: Pandas, Matplotlib, Sklearn, Pytorch, LightGBM, XGboost, Optuna, Github Actions, DVC, MLFlow, Airflow, SHAP, FastAPI.<br>
Metodology: CRISP-DM<br><br>

# Next Steps / Improvements

### Dataset Issues:
- Check some features issues: the deposit_type has the wrong values on this dataset ​​and it is certainly an important feature to determine whether or not the guest will cancel. The status_reservation_date could be broken down into other variables so that we would not lose the date the check-in was performed, which can be an important feature. This dataset also has too much duplicated values this strongly compromises the analysis.

### Know a bit more about Agencies and Companies:
- Get to know agencies and companies better. Perhaps a ranking system for these agencies and companies would benefit the machine learning model. With the current information we have, it is not possible to know with much criteria the quality of agencies and companies.

### Code:

- Substituir essa linha no github/workflows/ci.yaml - uses: iterative/setup-tools@v1  
- Configurar os parametros em um arquivo yaml
- Criar rotinas de teste CI/CD
- Criar um pipiline de preprocessanmento
- Optuna arredondar os valores diretamente
- Criar uma imagem Docker para o projeto
