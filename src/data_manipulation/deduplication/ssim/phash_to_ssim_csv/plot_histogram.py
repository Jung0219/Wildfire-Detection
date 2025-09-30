import pandas as pd
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

def plot_ssim_histogram(csv_path):
    # Load CSV
    df = pd.read_csv(csv_path)

    # Extract SSIM values
    ssim_values = df["ssim"]

    # Plot histogram
    plt.figure(figsize=(8, 6))
    plt.hist(ssim_values, bins=50, range=(0.0, 1.0), color='gray', edgecolor='black')
    plt.title("SSIM Histogram")
    plt.xlabel("SSIM Value")
    plt.ylabel("Count")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()

    # Save plot with same base name as CSV
    output_image = Path(csv_path).with_suffix('.png')
    plt.savefig(output_image)
    plt.close()
    print(f"Histogram saved to {output_image}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot SSIM histogram from CSV.")
    parser.add_argument("csv_path", type=str, help="Path to the CSV file containing SSIM results")

    args = parser.parse_args()

    plot_ssim_histogram(args.csv_path)
