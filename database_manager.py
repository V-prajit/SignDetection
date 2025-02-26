import numpy as np
import json
import os
from typing import List, Dict

class SignDatabase:
    def __init__(self, data_dir: str = "sign_database"):
        self.data_dir = data_dir
        self.signs = []
        self.sign_info = {}
        
        print(f"Initializing SignDatabase from {data_dir}")
        
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            print(f"Created database directory: {data_dir}")
        
        json_file = os.path.join(data_dir, "sign_data.json")
        if os.path.exists(json_file):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    signs_dict = data.get("signs", {})
                    signs_count = len(signs_dict)
                    print(f"Found {signs_count} signs in {json_file}")
                    
                    for path, sign_data in signs_dict.items():
                        sign_id = len(self.signs)
                        self.sign_info[sign_id] = {
                            "name": sign_data.get("name", "unknown"),
                            "path": path,
                            "isOneHanded": sign_data.get("is_one_handed", True),
                            "duration": sign_data.get("duration", 0),
                            "features": sign_data.get("features", {})
                        }
                        self.signs.append(sign_id)
                    
                    print(f"Loaded {len(self.signs)} signs from JSON database")
            except Exception as e:
                print(f"Error loading {json_file}: {str(e)}")
            
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
        if "features" in self.sign_info[sign_id]:
            features = self.sign_info[sign_id]["features"]
            features["is_one_handed"] = self.sign_info[sign_id].get('isOneHanded', False)
            return features
        else:
            try:
                data = np.load(os.path.join(self.data_dir, f"sign_{sign_id}.npz"))
                return {
                    'centroids_dom_arr': data['centroids_dom'],
                    'centroids_nondom_arr': data['centroids_nondom'],
                    'l_delta_arr': data['l_delta'],
                    'is_one_handed': self.sign_info[sign_id].get('isOneHanded', False)
                }
            except Exception as e:
                print(f"Error loading sign {sign_id}: {str(e)}")
                return {}
        
    def load_all_signs(self) -> List[Dict]:
        return [self.load_sign(sign_id) for sign_id in self.signs]
        
    def get_sign_info(self, sign_id: int) -> Dict:
        return self.sign_info.get(sign_id, {})