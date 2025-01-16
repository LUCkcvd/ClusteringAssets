import os
import re
import matplotlib.pyplot as plt

log_root = "log"
output_dir = os.path.join(log_root, "trend_graphs")
os.makedirs(output_dir, exist_ok=True)

# Regex for directory names containing single parameters
param_pattern = re.compile(r"(?P<param>[due])_(?P<value>[\d.]+)")

# Function to parse cluster sizes from the file
def parse_cluster_sizes(file_path):
    sizes = []
    print(f"Parsing file: {file_path}")
    with open(file_path, 'r') as file:
        for line in file:
            match = re.match(r"Cluster \d+, size (\d+)", line.strip())
            if match:
                sizes.append(int(match.group(1)))
                if len(sizes) == 10:  # Only top 10 entries
                    break
    if not sizes:
        print(f"No valid cluster sizes found in {file_path}")
    return sizes

# Data storage for each parameter
data = {"d": {}, "u": {}, "e": {}}

# Traverse the directory tree and collect data
for root, dirs, files in os.walk(log_root):
    for file in files:
        if file == "JournalTitleClusters.txt":
            file_path = os.path.join(root, file)
            sizes = parse_cluster_sizes(file_path)
            if sizes:
                avg_size = sum(sizes) / len(sizes)
                
                # Extract the parameter and value from the directory name
                directory_name = os.path.basename(root)
                match = param_pattern.match(directory_name)
                if match:
                    param = match.group("param")  # 'd', 'u', or 'e'
                    value = float(match.group("value"))
                    print(f"Extracted: {param}={value} from directory '{directory_name}'")
                    
                    # Store the average size under the correct parameter
                    if value not in data[param]:
                        data[param][value] = []
                    data[param][value].append(avg_size)
                else:
                    print(f"No match for parameters in directory '{directory_name}'")

# Compute average cluster sizes for each parameter value
for param in data:
    for value in data[param]:
        data[param][value] = sum(data[param][value]) / len(data[param][value])

print(f"Final aggregated data: {data}")

# Function to plot trends
def plot_trend(x, y, xlabel, ylabel, title, save_path):
    plt.figure()
    plt.plot(x, y, marker='o')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

# Generate and save plots for each parameter
for param in data:
    if data[param]:
        x = sorted(data[param].keys())
        y = [data[param][val] for val in x]
        plot_trend(
            x=x,
            y=y,
            xlabel=f"{param.upper()} Parameter",
            ylabel="Average Cluster Size (Top 10)",
            title=f"Trend of Avg Cluster Size vs {param.upper()}",
            save_path=os.path.join(output_dir, f"trend_{param}.png"),
        )

