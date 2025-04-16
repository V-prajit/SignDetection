from py4j.java_gateway import JavaGateway
import numpy as np
import threading
import traceback
import time
from concurrent.futures import ThreadPoolExecutor

class SignMatcher:
    _instance = None
    _lock = threading.Lock()
    
    @classmethod
    def get_instance(cls):
        """Singleton pattern to ensure only one instance is created"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        # Create a single Java Gateway connection
        print("Initializing Java DTW Gateway (this should happen only once)")
        self.gateway = JavaGateway()
        self.dtw_server = self.gateway.entry_point
        
        # Feature weights as described in the paper
        self.f1 = 2.0  # dominant hand centroids
        self.f2 = 1.0  # non-dominant hand centroids
        self.f3 = 1.0  # hand distance deltas
        self.f4 = 0.5  # dominant hand orientation
        self.f5 = 0.5  # non-dominant hand orientation
        self.f6 = 0.5  # orientation delta
        
        # Weight for hand appearance similarity
        self.f_hand = 1.0
        
        # Cache frequently used array converters
        self._double_array_class = self.gateway.jvm.double
        
        # Number of worker threads for parallel processing
        self.num_threads = max(6, threading.active_count() * 2)
        
        print(f"Initializing Sign Matcher with {self.num_threads} threads")
        
        # Thread local storage for per-thread Java gateways
        self.thread_local = threading.local()
    
    def get_thread_gateway(self):
        """Get or create a thread-local Java gateway"""
        if not hasattr(self.thread_local, 'gateway'):
            # Create a new gateway for this thread
            self.thread_local.gateway = JavaGateway()
            self.thread_local.dtw_server = self.thread_local.gateway.entry_point
            self.thread_local.double_array_class = self.thread_local.gateway.jvm.double
        return self.thread_local.gateway, self.thread_local.dtw_server, self.thread_local.double_array_class

    def convert_for_java(self, sequence, gateway=None, double_array_class=None):
        """Convert numpy array or list to Java 2D array for DTW calculation"""
        if sequence is None or len(sequence) == 0:
            return None
        
        # Use thread-local gateway if none provided
        if gateway is None or double_array_class is None:
            gateway, _, double_array_class = self.get_thread_gateway()
        
        # Convert list to numpy array if needed
        if isinstance(sequence, list):
            sequence = np.array(sequence, dtype=np.float32)

        # Create a contiguous copy for faster access
        sequence = np.ascontiguousarray(sequence, dtype=np.float32)
        
        # Create Java array with optimized bulk transfer
        nrows, ncols = sequence.shape
        Double2DArray = gateway.new_array(double_array_class, nrows, ncols)
        
        # Efficiently copy data - flatten inner loop for performance
        for i in range(nrows):
            row = sequence[i]
            for j in range(ncols):
                val = row[j]
                Double2DArray[i][j] = 0.0 if np.isnan(val) else float(val)
                        
        return Double2DArray

    def process_sign_batch(self, query_sign, db_signs_batch):
        """Process a batch of signs using thread-local Java gateway"""
        # Get thread-local gateway
        gateway, dtw_server, double_array_class = self.get_thread_gateway()
        
        results = []
        
        # Convert query features once for this batch
        q_dom = None
        q_nondom = None
        q_delta = None
        q_orient_dom = None
        q_orient_nondom = None
        q_orient_delta = None
        
        if 'centroids_dom_arr' in query_sign:
            q_dom = self.convert_for_java(query_sign['centroids_dom_arr'], gateway, double_array_class)
            
        if not query_sign.get('is_one_handed', True):
            if 'centroids_nondom_arr' in query_sign:
                q_nondom = self.convert_for_java(query_sign['centroids_nondom_arr'], gateway, double_array_class)
            if 'l_delta_arr' in query_sign:
                q_delta = self.convert_for_java(query_sign['l_delta_arr'], gateway, double_array_class)
            if 'orientation_dom_arr' in query_sign:
                q_orient_dom = self.convert_for_java(query_sign['orientation_dom_arr'], gateway, double_array_class)
            if 'orientation_nondom_arr' in query_sign:
                q_orient_nondom = self.convert_for_java(query_sign['orientation_nondom_arr'], gateway, double_array_class)
            if 'orientation_delta_arr' in query_sign:
                q_orient_delta = self.convert_for_java(query_sign['orientation_delta_arr'], gateway, double_array_class)
        
        # Process each database sign in the batch
        for idx, db_sign in db_signs_batch:
            try:
                # Calculate motion distance
                motion_distance = 0.0
                feature_count = 0
                
                # Process dominant hand centroids
                if q_dom is not None and 'centroids_dom_arr' in db_sign:
                    x_dom = self.convert_for_java(db_sign['centroids_dom_arr'], gateway, double_array_class)
                    if x_dom is not None:
                        dist = dtw_server.calculateDTW(q_dom, x_dom)
                        motion_distance += self.f1 * dist
                        feature_count += 1
                
                # Process non-dominant hand features if both signs are two-handed
                if not query_sign.get('is_one_handed', True) and not db_sign.get('is_one_handed', True):
                    # Non-dominant hand centroids
                    if q_nondom is not None and 'centroids_nondom_arr' in db_sign:
                        x_nondom = self.convert_for_java(db_sign['centroids_nondom_arr'], gateway, double_array_class)
                        if x_nondom is not None:
                            dist = dtw_server.calculateDTW(q_nondom, x_nondom)
                            motion_distance += self.f2 * dist
                            feature_count += 1
                    
                    # Hand distance deltas
                    if q_delta is not None and 'l_delta_arr' in db_sign:
                        x_delta = self.convert_for_java(db_sign['l_delta_arr'], gateway, double_array_class)
                        if x_delta is not None:
                            dist = dtw_server.calculateDTW(q_delta, x_delta)
                            motion_distance += self.f3 * dist
                            feature_count += 1
                    
                    # Orientation features if available
                    if q_orient_dom is not None and 'orientation_dom_arr' in db_sign:
                        x_orient_dom = self.convert_for_java(db_sign['orientation_dom_arr'], gateway, double_array_class)
                        if x_orient_dom is not None:
                            dist = dtw_server.calculateDTW(q_orient_dom, x_orient_dom)
                            motion_distance += self.f4 * dist
                            feature_count += 1
                    
                    if q_orient_nondom is not None and 'orientation_nondom_arr' in db_sign:
                        x_orient_nondom = self.convert_for_java(db_sign['orientation_nondom_arr'], gateway, double_array_class)
                        if x_orient_nondom is not None:
                            dist = dtw_server.calculateDTW(q_orient_nondom, x_orient_nondom)
                            motion_distance += self.f5 * dist
                            feature_count += 1
                    
                    if q_orient_delta is not None and 'orientation_delta_arr' in db_sign:
                        x_orient_delta = self.convert_for_java(db_sign['orientation_delta_arr'], gateway, double_array_class)
                        if x_orient_delta is not None:
                            dist = dtw_server.calculateDTW(q_orient_delta, x_orient_delta)
                            motion_distance += self.f6 * dist
                            feature_count += 1
                
                # Calculate average motion distance
                if feature_count > 0:
                    motion_distance /= feature_count
                
                # Calculate hand appearance distance
                hand_distance = self.compute_hand_distance(query_sign, db_sign)
                
                # Combine distances as in paper equation 12
                total_distance = motion_distance + self.f_hand * hand_distance
                
                results.append((idx, total_distance))
                
            except Exception as e:
                print(f"Error comparing with sign {idx}: {str(e)}")
                traceback.print_exc()
        
        return results

    def find_matches(self, query_sign, database_signs, top_k=10):
        """Find top k matches for query sign using parallel processing with Java DTW"""
        start_time = time.time()
        
        # Pre-filter compatible signs (same handedness)
        compatible_signs = [
            (idx, db_sign) for idx, db_sign in enumerate(database_signs)
            if query_sign.get('is_one_handed', True) == db_sign.get('is_one_handed', True)
        ]
        
        print(f"Comparing with {len(compatible_signs)} compatible signs using {self.num_threads} threads")
        
        # Split database into batches for parallel processing
        batch_size = max(1, len(compatible_signs) // self.num_threads)
        batches = [
            compatible_signs[i:i + batch_size]
            for i in range(0, len(compatible_signs), batch_size)
        ]
        
        # Process batches in parallel
        distances = []
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            # Submit all batch processing jobs
            futures = [
                executor.submit(self.process_sign_batch, query_sign, batch)
                for batch in batches
            ]
            
            # Collect results as they complete
            for future in futures:
                try:
                    batch_results = future.result()
                    distances.extend(batch_results)
                except Exception as e:
                    print(f"Error processing batch: {str(e)}")
                    traceback.print_exc()
        
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
        
        time_taken = time.time() - start_time
        print(f"Matching completed in {time_taken:.2f}s")
        
        return matches[:top_k]

    # Add this method to the SignMatcher class in sign_matcher.py

    def find_matches_batch(self, query_sign, database_signs, top_k=10):
        """Find top k matches using batch processing for much faster results"""
        start_time = time.time()
        
        # Filter compatible signs (same handedness)
        compatible_signs = [
            db_sign for db_sign in database_signs
            if query_sign.get('is_one_handed', True) == db_sign.get('is_one_handed', True)
        ]
        
        print(f"Comparing with {len(compatible_signs)} compatible signs using batch processing")
        
        # Extract dominant hand centroids (main feature)
        query_centroids = None
        if 'centroids_dom_arr' in query_sign:
            query_centroids = np.array(query_sign['centroids_dom_arr'], dtype=np.float32)
            query_centroids = np.ascontiguousarray(query_centroids)
        else:
            print("No dominant hand centroids found in query")
            return []
        
        # Convert query to Java format once
        java_query = self.convert_for_java(query_centroids)
        
        # Create a Java ArrayList to hold the database sequences
        java_list = self.gateway.jvm.java.util.ArrayList()
        
        # Prepare all database sequences
        for db_sign in compatible_signs:
            if 'centroids_dom_arr' in db_sign:
                centroids = np.array(db_sign['centroids_dom_arr'], dtype=np.float32)
                centroids = np.ascontiguousarray(centroids)
                java_centroids = self.convert_for_java(centroids)
                java_list.add(java_centroids)
            else:
                # Use empty array as placeholder
                empty = np.zeros((1, 2), dtype=np.float32)
                java_list.add(self.convert_for_java(empty))
        
        # Call batch processing function
        print(f"Processing {java_list.size()} sequences in a single batch")
        
        # Pass the Java ArrayList to the batch function
        distances = self.dtw_server.batchCalculateDTW(java_query, java_list)
        
        # Process results
        matches = []
        for i, distance in enumerate(distances):
            matches.append((i, distance))
        
        # Sort by distance (ascending)
        matches.sort(key=lambda x: x[1])
        
        # Convert to similarity scores (0-100%)
        result_matches = []
        if matches:
            min_dist = matches[0][1]
            max_dist = matches[-1][1]
            dist_range = max_dist - min_dist
            
            for idx, dist in matches[:top_k]:
                if dist_range > 0:
                    similarity = (1.0 - (dist - min_dist) / dist_range) * 100
                else:
                    similarity = 100.0
                result_matches.append((idx, similarity))
        
        print(f"DTW batch processing completed in {time.time() - start_time:.2f} seconds")
        
        return result_matches

    def compute_hand_distance(self, Q, X):
        """Compute Euclidean distance between hand appearance images as in paper section 6"""
        try:
            # Per equation 11 in the paper - use Euclidean distance between hand images
            total_distance = 0.0
            
            # Dominant hand at start frame
            if 'H_d_s' in Q and 'H_d_s' in X:
                Q_d_s = np.asarray(Q['H_d_s'], dtype=np.float32).ravel()
                X_d_s = np.asarray(X['H_d_s'], dtype=np.float32).ravel()
                total_distance += np.linalg.norm(Q_d_s - X_d_s)
                
            # Dominant hand at end frame
            if 'H_d_e' in Q and 'H_d_e' in X:
                Q_d_e = np.asarray(Q['H_d_e'], dtype=np.float32).ravel()
                X_d_e = np.asarray(X['H_d_e'], dtype=np.float32).ravel()
                total_distance += np.linalg.norm(Q_d_e - X_d_e)
            
            # Non-dominant hand if applicable
            if not Q.get('is_one_handed', True) and not X.get('is_one_handed', True):
                if 'H_nd_s' in Q and 'H_nd_s' in X:
                    Q_nd_s = np.asarray(Q['H_nd_s'], dtype=np.float32).ravel()
                    X_nd_s = np.asarray(X['H_nd_s'], dtype=np.float32).ravel()
                    total_distance += np.linalg.norm(Q_nd_s - X_nd_s)
                    
                if 'H_nd_e' in Q and 'H_nd_e' in X:
                    Q_nd_e = np.asarray(Q['H_nd_e'], dtype=np.float32).ravel()
                    X_nd_e = np.asarray(X['H_nd_e'], dtype=np.float32).ravel()
                    total_distance += np.linalg.norm(Q_nd_e - X_nd_e)
                    
            return total_distance
            
        except Exception as e:
            print(f"Error computing hand distance: {str(e)}")
            traceback.print_exc()
            return float('inf')