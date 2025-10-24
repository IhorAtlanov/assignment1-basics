import pickle

file_path = '/mnt/d/Stanford_LLM/assignment1-basics/cs336_basics/BPE/bpe_model_owt_train.pkl'

with open(file_path, 'rb') as file:
    data = pickle.load(file)

print(data)