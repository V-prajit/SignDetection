import json

def rank_signs(metric="python_dtw"):
    with open("benchmark_results.json", "r") as file:
        results = json.load(file)

    if metric not in results[0]:
        print(f"Invalid metric: {metric}. Choose from: python_dtw, python_fastdtw, java_dtw, java_fastdtw")
        return

    ranked = sorted(results, key=lambda x: x[metric])

    print(f"\n--- Ranked Sign Similarities by {metric} ---")
    for rank, entry in enumerate(ranked, 1):
        print(f"{rank}. {entry['sign']} - Distance: {entry[metric]:.4f}")

if __name__ == "__main__":
    print("Choose a ranking metric: python_dtw, python_fastdtw, java_dtw, java_fastdtw")
    metric = input("Enter metric: ").strip()
    rank_signs(metric)
