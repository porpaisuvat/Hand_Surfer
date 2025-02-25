import csv
import os

# ✅ Get the absolute path of the `ANN/` folder
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "ANN", "data"))

def get_file_name():
    """Ask the user for a CSV file name before saving data."""
    file_name = input("Enter the file name (without .csv): ").strip()
    if not file_name:
        file_name = "pose_data"  # Default name if empty
    return os.path.join(BASE_DIR, file_name + ".csv")  # ✅ Save to `ANN/data/` folder

def write_csv(file_path, header, data):
    """Ensure 'ANN/data/' exists and write pose data to CSV file."""
    
    # ✅ Ensure the directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)  

    # ✅ Now write to the file
    with open(file_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)
    print(f"✅ Data successfully saved to '{file_path}'")
