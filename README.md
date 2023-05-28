[português (BR)](https://github.com/alyssonvidal/Hotel-Booking-Cancelations/blob/main/README_PT.md)

<center><img src="/images/hotel.jpg" alt="logo" width="800" height="600"/></center>

# Problem Statement

One of the common main problems the hotels facing are cancellations. There are several problems due to cancellations, including: operational problems, replanning of tasks, inaccuracy in revenue forecasts, poor optimization of resources such as occupancy of rooms..

This study contains informations about a Ciy Hotel and a Resort Hotel in the period between July 2015 and August 2017, the City Hotel is located in Lisbon, capital of Portugal and the Resort Hotel in Algarve, a coastal city also from Portugal.

[glossário](https://github.com/alyssonvidal/Hotel-Booking-Cancelations/blob/main/referenses/glossary_PT.md)

# Objective

* Develop a machine learning model capable of predicting which customers are most likely to cancel a reservation
* Identify which are the main features available in the database that identify whether the customer will cancel or not.
* Present a possible business plan based on the results obtained.<br><br>

# Development Stages
[**Data Preprocessing**](https://github.com/alyssonvidal/Bank-Marketing-Cluster/blob/main/notebooks/part01_preprocessing.ipynb)<br>
Loading Data, Dealing with Missing Values, Dealing with Duplicated Values, Dealing with Strange Values, Fixing Data Types, Featrue Engieneering.

[**Exploratory Data Analysis**](https://github.com/alyssonvidal/Bank-Marketing-Cluster/blob/main/notebooks/part02_eda.ipynb)<br>
Mind Map Business, Hypotesis, Summary Table, Bivariated Analysis over Cancelations, Multivariated Analysis.

[**Data Preparation**](https://github.com/alyssonvidal/Bank-Marketing-Cluster/blob/main/notebooks/part03_model.ipynb)<br>
Encoding, Normalization, Feature Selection.

[**Modelo de Machine Learning**](https://github.com/alyssonvidal/Bank-Marketing-Cluster/blob/main/notebooks/part03_model.ipynb)<br>
Models: GBM, XGB, Pytorch.

[**Hypertuning**](https://github.com/alyssonvidal/Bank-Marketing-Cluster/blob/main/notebooks/part03_model.ipynb)<br>
Optuna.

[**Feature Importance**](https://github.com/alyssonvidal/Bank-Marketing-Cluster/blob/main/notebooks/part03_model.ipynb)<br>
Filter Methods - Anova, Chi2, Embeeded Methods - LGBM, XGboost, Wrapper Methods - Shapley Values<br><br>

# Reports
[**Overall Report**](https://github.com/alyssonvidal/Bank-Marketing-Cluster/blob/main/reports/resultados.md)<br>
[**Last Month Report**](https://github.com/alyssonvidal/Bank-Marketing-Cluster/blob/main/reports/resultados.md)<br>
[**Model Report**](https://github.com/alyssonvidal/Bank-Marketing-Cluster/blob/main/reports/resultados.md)<br><br>

# Tools
Languages: Python<br>
IDE: Visual Studio Code, Jupyter Notebook<br>
Frameworks: Pandas, Matplotlib, Seaborn, Sklearn, Pytorch<br>
Metodology: CRISP-DM<br><br>

# Next Steps
- Check some features issues: the deposit_type has the wrong values on this dataset ​​and it is certainly an important feature to determine whether or not the guest will cancel. The status_reservation_date could be broken down into other variables so that we would not lose the date the check-in was performed, which can be an important feature.

- Get to know agencies and companies better. Perhaps a ranking system for these agencies and companies would benefit the machine learning model. With the current information we have, it is not possible to know with much criteria the quality of agencies and companies.


