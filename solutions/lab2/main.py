from prettytable import PrettyTable

# Initialize the table
table = PrettyTable()
table.field_names = ["Name", "Age", "Profession"]

# Add rows
table.add_row(["Alice", 24, "Engineer"])
table.add_row(["Bob", 27, "Doctor"])
table.add_row(["Charlie", 22, "Artist"])

# Print the table
print(table)
