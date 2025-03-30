from py4j.java_gateway import JavaGateway
import numpy as np
from scipy.spatial.distance import euclidean

class SignMatcher:
    def __init__(self):
        self.gateway = JavaGateway()
        self.dtw_server = self.gateway.entry_point
        
        self.f1 = 2.0 
        self.f2 = 1.0  
        self.f3 = 1.0  
        self.f4 = 0.5  
        self.f5 = 0.5 
        self.f6 = 0.5  
        
        self.f_hand = 1.0

    def convert_for_java(self, sequence):
        if sequence is None or len(sequence) == 0:
            return None
        
        if isinstance(sequence, list):
            sequence = np.array(sequence, dtype=np.float32)

        nrows, ncols = sequence.shape
        Double2DArray = self.gateway.new_array(self.gateway.jvm.double, nrows, ncols)
        
        for i in range(nrows):
            for j in range(ncols):
                val = sequence[i][j]
                if np.isnan(val):
                    Double2DArray[i][j] = 0.0
                else:
                    Double2DArray[i][j] = float(val)
                    
        return Double2DArray

    def compute_dtw_distance(self, Q, X):
        try:
            motion_distance = 0.0
            feature_count = 0
            
            if 'centroids_dom_arr' in Q and 'centroids_dom_arr' in X:
                q_dom = self.convert_for_java(Q['centroids_dom_arr'])
                x_dom = self.convert_for_java(X['centroids_dom_arr'])
                if q_dom is not None and x_dom is not None:
                    dist = self.dtw_server.computeFastDTW(q_dom, x_dom, 1)
                    motion_distance += self.f1 * dist
                    feature_count += 1

            if not Q.get('is_one_handed', True) and not X.get('is_one_handed', True):
                if 'centroids_nondom_arr' in Q and 'centroids_nondom_arr' in X:
                    q_nondom = self.convert_for_java(Q['centroids_nondom_arr'])
                    x_nondom = self.convert_for_java(X['centroids_nondom_arr'])
                    if q_nondom is not None and x_nondom is not None:
                        dist = self.dtw_server.computeFastDTW(q_nondom, x_nondom, 1)
                        motion_distance += self.f2 * dist
                        feature_count += 1

                if 'l_delta_arr' in Q and 'l_delta_arr' in X:
                    q_delta = self.convert_for_java(Q['l_delta_arr'])
                    x_delta = self.convert_for_java(X['l_delta_arr'])
                    if q_delta is not None and x_delta is not None:
                        dist = self.dtw_server.computeFastDTW(q_delta, x_delta, 1)
                        motion_distance += self.f3 * dist
                        feature_count += 1

            if feature_count > 0:
                motion_distance /= feature_count

            return motion_distance

        except Exception as e:
            print(f"Error computing DTW distance: {str(e)}")
            traceback.print_exc()
            return float('inf')

    def find_matches(self, query_sign, database_signs, top_k=10):
        distances = []
        
        for idx, db_sign in enumerate(database_signs):
            try:
                # Skip signs with different handedness
                if query_sign.get('is_one_handed', True) != db_sign.get('is_one_handed', True):
                    continue
                    
                # Calculate motion and hand distances
                motion_dist = self.compute_dtw_distance(query_sign, db_sign)
                hand_dist = self.compute_hand_distance(query_sign, db_sign)
                
                # Combine distances using the hand weight factor
                total_dist = motion_dist + self.f_hand * hand_dist
                distances.append((idx, total_dist))
                
            except Exception as e:
                print(f"Error comparing with sign {idx}: {str(e)}")
                continue
        
        if not distances:
            return []
            
        # Find min and max distances for normalization
        min_dist = min(d for _, d in distances)
        max_dist = max(d for _, d in distances)
        dist_range = max_dist - min_dist
        
        matches = []
        for idx, dist in distances:
            if dist_range > 0:
                # Linear normalization to convert distance to similarity (0-100%)
                normalized_dist = (dist - min_dist) / dist_range
                similarity = (1.0 - normalized_dist) * 100
            else:
                # Handle case where all distances are equal
                similarity = 100.0 if dist == min_dist else 0.0
                
            matches.append((idx, similarity))
        
        # Sort by similarity (descending) and return top_k matches
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches[:top_k]

    def compute_hand_distance(self, Q, X):
        try:
            Q_d_s = np.asarray(Q['H_d_s'], dtype=np.float32).ravel()
            X_d_s = np.asarray(X['H_d_s'], dtype=np.float32).ravel()
            Q_d_e = np.asarray(Q['H_d_e'], dtype=np.float32).ravel()
            X_d_e = np.asarray(X['H_d_e'], dtype=np.float32).ravel()
            
            d_start = np.linalg.norm(Q_d_s - X_d_s)
            d_end = np.linalg.norm(Q_d_e - X_d_e)
            
            total_distance = d_start + d_end
            
            if not Q.get('is_one_handed', True) and not X.get('is_one_handed', True):
                if 'H_nd_s' in Q and 'H_nd_e' in Q and 'H_nd_s' in X and 'H_nd_e' in X:
                    Q_nd_s = np.asarray(Q['H_nd_s'], dtype=np.float32).ravel()
                    X_nd_s = np.asarray(X['H_nd_s'], dtype=np.float32).ravel()
                    Q_nd_e = np.asarray(Q['H_nd_e'], dtype=np.float32).ravel()
                    X_nd_e = np.asarray(X['H_nd_e'], dtype=np.float32).ravel()
                    
                    nd_start = np.linalg.norm(Q_nd_s - X_nd_s)
                    nd_end = np.linalg.norm(Q_nd_e - X_nd_e)
                    total_distance += nd_start + nd_end
            
            return total_distance
        except Exception as e:
            print(f"Error computing hand distance: {str(e)}")
            traceback.print_exc()
            return float('inf')

    def compare_signs(self, Q, X):
        try:
            dtw_distance = 0.0
            
            if Q['centroids_dom_arr'] and X['centroids_dom_arr']:
                q_dom = self.convert_for_java(Q['centroids_dom_arr'])
                x_dom = self.convert_for_java(X['centroids_dom_arr'])
                if q_dom is not None and x_dom is not None:
                    dist = self.dtw_server.computeFastDTW(q_dom, x_dom, 1)
                    dtw_distance += self.f1 * dist
            
            if not Q.get('is_one_handed', True) and 'centroids_nondom_arr' in Q:
                q_nondom = self.convert_for_java(Q['centroids_nondom_arr'])
                x_nondom = self.convert_for_java(X['centroids_nondom_arr'])
                if q_nondom is not None and x_nondom is not None:
                    dist = self.dtw_server.computeFastDTW(q_nondom, x_nondom, 1)
                    dtw_distance += self.f2 * dist
                
                if Q['l_delta_arr'] and X['l_delta_arr']:
                    q_delta = self.convert_for_java(Q['l_delta_arr'])
                    x_delta = self.convert_for_java(X['l_delta_arr'])
                    if q_delta is not None and x_delta is not None:
                        dist = self.dtw_server.computeFastDTW(q_delta, x_delta, 1)
                        dtw_distance += self.f3 * dist
            
            for feature, weight in [
                ('orientation_dom_arr', self.f4),
                ('orientation_nondom_arr', self.f5),
                ('orientation_delta_arr', self.f6)
            ]:
                if feature in Q and feature in X:
                    q_orient = self.convert_for_java(Q[feature])
                    x_orient = self.convert_for_java(X[feature])
                    if q_orient is not None and x_orient is not None:
                        dist = self.dtw_server.computeFastDTW(q_orient, x_orient, 1)
                        dtw_distance += weight * dist
            
            hand_distance = self.compute_hand_distance(Q, X)
            
            total_distance = dtw_distance + self.f_hand * hand_distance
            
            return total_distance
            
        except Exception as e:
            print(f"Error comparing signs: {str(e)}")
            return float('inf')