from py4j.java_gateway import JavaGateway

gateway = JavaGateway()
dtw_server = gateway.entry_point

def convert_to_java_array(py_array):
    nrows = len(py_array)
    ncols = len(py_array[0])

    Double2DArray = gateway.new_array(gateway.jvm.double, nrows, ncols)

    for i in range(nrows):
        for j in range(ncols):
            Double2DArray[i][j] = float(py_array[i][j])

    return Double2DArray

def run_comparison_tests():
    from signPatterns import generate_sign_patterns
    import time
    import json
    from DTWNormal import standard_dtw
    from DTWNormal import standard_dtw as DTW_Distance

    signs = generate_sign_patterns()
    results = []

    for sign_name, sign_data in signs.items():
        print(f"\nAnalyzing '{sign_name}' sign:")
        reference = convert_to_java_array(sign_data['reference'])

        for idx, variation in enumerate(sign_data['variations'], 1):
            print(f"\nTesting Variation {idx} of '{sign_name}'")

            variation_java = convert_to_java_array(variation)

            start_time = time.time()
            std_distance = standard_dtw(sign_data['reference'], variation)
            std_time = time.time() - start_time

            start_time = time.time()
            fast_distance = DTW_Distance(sign_data['reference'], variation)
            fast_time = time.time() - start_time

            start_time = time.time()
            java_dtw_distance = dtw_server.calculateDTW(reference, variation_java)
            java_dtw_time = time.time() - start_time

            start_time = time.time()
            java_fast_dtw_distance = dtw_server.computeFastDTW(reference, variation_java, 1)
            java_fast_time = time.time() - start_time

            print(f"Standard DTW (Python)    - Distance: {std_distance:.4f} | Time: {std_time:.4f}s")
            print(f"FastDTW (Python)         - Distance: {fast_distance:.4f} | Time: {fast_time:.4f}s")
            print(f"DTW (Java via Py4J)      - Distance: {java_dtw_distance:.4f} | Time: {java_dtw_time:.4f}s")
            print(f"FastDTW (Java via Py4J)  - Distance: {java_fast_dtw_distance:.4f} | Time: {java_fast_time:.4f}s")

            results.append({
                "sign": f"{sign_name}_var{idx}",
                "python_dtw": std_distance,
                "python_fastdtw": fast_distance,
                "java_dtw": java_dtw_distance,
                "java_fastdtw": java_fast_dtw_distance
            })

    with open("benchmark_results.json", "w") as file:
        json.dump(results, file, indent=4)

if __name__ == "__main__":
    run_comparison_tests()
    print("\nBenchmarking completed. Results saved to 'benchmark_results.json'. Run 'rank.py' to rank them.")
