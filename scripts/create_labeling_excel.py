import pandas as pd
from pathlib import Path
from openpyxl import load_workbook
from openpyxl.worksheet.datavalidation import DataValidation

csv_file = Path("data/golden/intent_labeling.csv")
xlsx_file = Path("data/golden/intent_labeling.xlsx")

labels = [
    "ios_update_issues",
    "apps_services",
    "battery_power",
    "device_performance",
    "hardware_accessories",
    "keyboard_input",
    "connectivity",
    "photos_data",
    "apple_id_account",
    "purchases_orders_refunds",
    "other_unclear"
]

df = pd.read_csv(csv_file, dtype=str).fillna("")
df.to_excel(xlsx_file, index=False)

wb = load_workbook(xlsx_file)
ws = wb.active

# Dropdown for intent_label (column D)
dv = DataValidation(
    type="list",
    formula1='"' + ",".join(labels) + '"',
    allow_blank=True
)

ws.add_data_validation(dv)
dv.add("D2:D201")

# Make the sheet easier to read
ws.column_dimensions["A"].width = 22
ws.column_dimensions["B"].width = 70
ws.column_dimensions["C"].width = 70
ws.column_dimensions["D"].width = 28
ws.column_dimensions["E"].width = 25

ws.freeze_panes = "A2"
ws.auto_filter.ref = ws.dimensions

wb.save(xlsx_file)

print("Created:")
print(xlsx_file)
print()
print("Open this file in Excel and use the dropdown in column D.")
print("Your first 5 labels are preserved.")
