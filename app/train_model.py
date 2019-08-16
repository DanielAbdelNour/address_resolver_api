import os
from fasttext import FastText
from sklearn.metrics import pairwise_distances
import pandas as pd
import numpy as np
import jellyfish
import tqdm
from tqdm import tqdm
from pathlib import Path
import ipdb

if Path.cwd().name == 'app':
    base_path = Path('.')
else:
    base_path = Path('app')
    
files_path = base_path / 'files/'

gnaf_addresses = pd.read_csv(files_path / 'gnaf_addresses.csv', low_memory=False)
concat_address = pd.read_csv(files_path / 'address_clean.txt', header=None)[0]

mdl = FastText.train_unsupervised(
    input=str(files_path / 'address_clean.txt'),
    minCount=0,
    minn=0,
    maxn=3,
    dim=300,
    epoch=10,
    bucket=200000
)
mdl.save_model(str(files_path / 'address_resolver.mdl'))

address_vecs = [mdl.get_sentence_vector(addr) for addr in tqdm(concat_address.values)]
np.save(files_path / 'address_vecs.npy', address_vecs)


raw_address='8mayfrd avehopevally'
raw_address='55 curry st adelaid'
raw_address=raw_address.upper()
raw_address_vec = mdl.get_sentence_vector(raw_address)        
distances = pairwise_distances([raw_address_vec], address_vecs)        
closest = np.argsort(distances)[0][0:100]        
local_closest_str = concat_address[closest].values        

#str_dist = [jellyfish.jaro_distance(raw_address, x) for x in local_closest_str]  
#local_closest_idx = np.argsort(1-np.array(str_dist))[0]      

str_dist = [jellyfish.levenshtein_distance(raw_address, x) for x in local_closest_str]  
local_closest_idx = np.argsort(np.array(str_dist))[0]      

global_closest_idx = closest[local_closest_idx]        
gnaf_address_match = concat_address.iloc[global_closest_idx]    
print(gnaf_address_match)
  

# matche_dists = [1-jellyfish.jaro_winkler(raw_address, x) for x in concat_address.values]
# best_matches = np.argsort(matche_dists)
# print(concat_address[best_matches[0]])
