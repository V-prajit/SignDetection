from signPatterns import generate_sign_patterns
from DTWNormal import standard_dtw
from FastDTW import DTW_Distance
import time
from py4j.java_gateway import JavaGateway

gateway = JavaGateway();
dtw_server = gateway.entry_point

def run_comparison_tests():
    signs = generate_sign_patterns()

    if not signs: 
        print("No sign patterns were generated. Please check signPatterns.py")
        return

    print("Sign Language Pattern Recognition Comparison\n")
    print("Testing both DTW algorithms on various sign patterns...")

    for sign_name, sign_data in signs.items():
        print(f"\nAnalyzing '{sign_name}' sign:")
        reference = sign_data['reference']
        
        if reference is None or len(reference) == 0:
            print(f"Warning: Reference pattern for {sign_name} is empty")
            continue

        for idx, variation in enumerate(sign_data['variations'], 1):
            if variation is None or len(variation) == 0:
                print(f"Warning: Variation {idx} for {sign_name} is empty")
                continue

            print(f"\nTesting Variation {idx}:")
            
            start_time = time.time()
            std_distance = standard_dtw(reference, variation)
            std_time = time.time() - start_time

            start_time = time.time()
            fast_distance = DTW_Distance(reference, variation)
            fast_time = time.time() - start_time

            print(f"Standard DTW:")
            print(f"  - Time: {std_time:.4f} seconds")
            print(f"  - Distance: {std_distance:.4f}")
            
            print(f"FastDTW:")
            print(f"  - Time: {fast_time:.4f} seconds")
            print(f"  - Distance: {fast_distance:.4f}")
            
            print(f"Comparison:")
            print(f"  - Speed improvement: {std_time/fast_time:.2f}x faster")
            print(f"  - Distance difference: {abs(std_distance - fast_distance):.6f}")

            
if __name__ == "__main__":
    run_comparison_tests()