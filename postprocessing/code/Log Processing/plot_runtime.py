import os
import re
import matplotlib.pyplot as plt

log_root = "log"
output_dir = os.path.join(log_root, "trend_graphs")
os.makedirs(output_dir, exist_ok=True)

# Regex for directory names containing single parameters
param_pattern = re.compile(r"(?P<param>[due])_(?P<value>[\d.]+)")

# Regex for extracting 'sys' runtime from runtime.log
runtime_pattern = re.compile(r"sys\s+(\d+)m([\d.]+)s")

# Function to parse the sys runtime from runtime.log
def parse_runtime(file_path):
    print(f"Parsing runtime from file: {file_path}")
    with open(file_path, 'r') as file:
        for line in file:
            match = runtime_pattern.search(line.strip())
            if match:
                minutes = int(match.group(1))
                seconds = float(match.group(2))
                runtime = minutes * 60 + seconds  # Convert to total seconds
                return runtime
    print(f"No valid runtime found in {file_path}")
    return None

# Data storage for each parameter
data = {"d": {}, "u": {}, "e": {}}

# Traverse the directory tree and collect data
for root, dirs, files in os.walk(log_root):
    for file in files:
        if file == "runtime.log":
            file_path = os.path.join(root, file)
            runtime = parse_runtime(file_path)
            if runtime is not None:
                # Extract the parameter and value from the directory name
                directory_name = os.path.basename(root)
                match = param_pattern.match(directory_name)
                if match:
                    param = match.group("param")  # 'd', 'u', or 'e'
                    value = float(match.group("value"))
                    print(f"Extracted: {param}={value} with runtime={runtime} from directory '{directory_name}'")
                    
                    # Store the runtime under the correct parameter
                    if value not in data[param]:
                        data[param][value] = []
                    data[param][value].append(runtime)
                else:
                    print(f"No match for parameters in directory '{directory_name}'")

# Compute average runtime for each parameter value
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
            ylabel="Runtime (seconds)",
            title=f"Trend of Runtime vs {param.upper()}",
            save_path=os.path.join(output_dir, f"trend_runtime_{param}.png"),
        )

