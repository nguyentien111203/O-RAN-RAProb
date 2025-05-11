from utils.load_data import load_env_params_from_csv
from train.train import train_model

input_file = 'PPOtrain/data/input_file.csv'
env_params_list = load_env_params_from_csv(input_file)
train_model(env_params_list, total_epochs=500)
