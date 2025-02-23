import numpy as np
import json
import os
from typing import List, Dict

class SignDatabase:
    def __init__(self, data_dir: str = "sign_database"):
        self.data_dir = data_dir
        self.signs = []
        self.sign_info = {}
        
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            
    def add_sign(self, sign_features: Dict, sign_info: Dict):
        sign_id = len(self.signs)
        
        np.savez(
            os.path.join(self.data_dir, f"sign_{sign_id}.npz"),
            centroids_dom=sign_features['centroids_dom_arr'],
            centroids_nondom=sign_features['centroids_nondom_arr'],
            l_delta=sign_features['l_delta_arr']
        )
        
        self.sign_info[sign_id] = sign_info
        self.signs.append(sign_id)

        with open(os.path.join(self.data_dir, "sign_info.json"), "w") as f:
            json.dump(self.sign_info, f)
            
    def load_sign(self, sign_id: int) -> Dict:
        data = np.load(os.path.join(self.data_dir, f"sign_{sign_id}.npz"))
        return {
            'centroids_dom_arr': data['centroids_dom'],
            'centroids_nondom_arr': data['centroids_nondom'],
            'l_delta_arr': data['l_delta'],
            'isOneHanded': self.sign_info[sign_id].get('isOneHanded', False)
        }
        
    def load_all_signs(self) -> List[Dict]:
        return [self.load_sign(sign_id) for sign_id in self.signs]
        
    def get_sign_info(self, sign_id: int) -> Dict:
        return self.sign_info.get(sign_id, {})