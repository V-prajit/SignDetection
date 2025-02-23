from sign_matcher import SignMatcher
from database_manager import SignDatabase
import numpy as np

def create_test_sign():
    return {
        'centroids_dom_arr': np.random.rand(20, 2),
        'centroids_nondom_arr': np.random.rand(20, 2),
        'l_delta_arr': np.random.rand(20, 2),
        'isOneHanded': False
    }

def test_database_operations():
    print("Testing database operations...")
    db = SignDatabase("test_database")
    
    test_sign = create_test_sign()
    db.add_sign(test_sign, {'name': 'test_sign_1'})
    
    loaded_sign = db.load_sign(0)
    print("Successfully saved and loaded sign from database")
    return db

def test_sign_matching():
    print("Testing sign matching...")
    
    matcher = SignMatcher()
    
    query_sign = create_test_sign()
    database_signs = [create_test_sign() for _ in range(5)]
    
    matches = matcher.find_matches(query_sign, database_signs, top_k=3)
    print(f"Found {len(matches)} matches")
    for idx, distance in matches:
        print(f"Match {idx}: distance = {distance}")

def main():
    print("Starting tests...")
    
    try:
        db = test_database_operations()
        
        test_sign_matching()
        
        print("All tests completed successfully!")
        
    except Exception as e:
        print(f"Test failed: {str(e)}")

if __name__ == "__main__":
    main()