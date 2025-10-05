import csv
import json

csv_file = "Cap3D_automated_ShapeNet.csv"
json_file = "Cap3D_automated_ShapeNet.json"

data_dict = {}

with open(csv_file, newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    for row in reader:
        if len(row) >= 2:
            key = row[0].strip()
            value = row[1].strip()
            data_dict[key] = value

with open(json_file, "w", encoding='utf-8') as f:
    json.dump(data_dict, f, indent=2, ensure_ascii=False)

print(f"JSON file saved as {json_file}")