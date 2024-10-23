import os

# Step 1: Preprocess the data
os.system('python src/data_preprocessing.py')

# Step 2: Train the models
os.system('python src/model_training.py')

# Step 3: Make predictions
os.system('python src/make_predictions.py')
