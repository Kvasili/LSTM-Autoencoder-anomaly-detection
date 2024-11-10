import pandas as pd

# Read the CSV file into a DataFrame
# Replace with your input file name
input_filename = './data/CAM_sensor_mixed_data.csv'
df = pd.read_csv(input_filename)

# Select specific columns to save in the new CSV
# Replace with your column names
selected_columns = ["index", "nfd-1-cps", "nfd-1-cr", "rr-active-state", "rr-position",
                    "ss1-active-state", "ss1-position", "ss2-active-state", "ss2-position", "manual-scram"]

selected_columns = ["CAM-CNT", "NFD-1-CPS", "cam-cnt_abnormal"]  #
df_selected = df[selected_columns]

# Save the selected columns to a new CSV file
# Replace with your output file name
output_filename = './CAM_sensor_mixed_data.csv'
df_selected.to_csv(output_filename, index=False)

print(f"New CSV with selected columns saved as '{output_filename}'")
