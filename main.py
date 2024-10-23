import os

os.system('python src/data_preprocessing.py')

os.system('python src/feature_engineering.py')

os.system('python src/model_training.py')

os.system('streamlit run bus_demand_app.py')
