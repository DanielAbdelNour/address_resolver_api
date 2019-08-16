from fasttext import FastText
from sklearn.metrics import pairwise_distances
import pandas as pd
import numpy as np
from flask import Flask, jsonify
from flask_restful import reqparse, abort, Api, Resource
import json
import jellyfish
from pathlib import Path

app = Flask(__name__)
api = Api(app)

parser = reqparse.RequestParser()
parser.add_argument('raw_address')

if Path.cwd().name == 'app':
    base_path = Path('.')
else:
    base_path = Path('app')
    
files_path = base_path / 'files/'

gnaf_addresses = pd.read_csv(files_path / 'gnaf_addresses.csv', low_memory=False)
mdl = FastText.load_model(str(files_path / 'address_resolver.mdl'))
address_vecs = np.load(files_path / 'address_vecs.npy')
concat_address = pd.read_csv(files_path / 'address_clean.txt', header=None)[0]


class AddressResolver(Resource):
    def raw2gnaf(self, raw_address):
        raw_address_vec = mdl.get_sentence_vector(raw_address)
        distances = pairwise_distances([raw_address_vec], address_vecs)
        closest = np.argsort(distances)[0][0:100]
        local_closest_str = concat_address[closest].values
        jaro_dist = [jellyfish.levenshtein_distance(raw_address, x) for x in local_closest_str]
        local_closest_idx = np.argsort(np.array(jaro_dist))[0]
        global_closest_idx = closest[local_closest_idx]
        gnaf_address_match = json.loads(gnaf_addresses.iloc[global_closest_idx].to_json())
        return gnaf_address_match

        
    def get(self):
        try:
            args = parser.parse_args()
            raw_address = args['raw_address'].upper()
            gnaf_address_match = self.raw2gnaf(raw_address)
            return gnaf_address_match, 201
        except:
            return 'Address Resolver API', 201


    def post(self):
        try:
            args = parser.parse_args()
            raw_address = args['raw_address'].upper()
            gnaf_address_match = self.raw2gnaf(raw_address)
            return gnaf_address_match, 201
        except:
            return 'failed', 404



api.add_resource(AddressResolver, '/')



# if __name__ == '__main__':
#     app.run(host="0.0.0.0", port=5555, debug=True)
