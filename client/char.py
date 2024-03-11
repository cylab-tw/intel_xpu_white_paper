import os
import shutil
import json
import matplotlib.pyplot as plt

def Average(lst):
    return str(round(sum(lst) / len(lst), 3))

def generate_comparison_graphs(data_dir, output_filename):
    # Load JSON files
    with open(data_dir + "stock.json", "r") as f:
        train_info_1 = json.load(f)

    with open(data_dir + "intel.json", "r") as f:
        train_info_2 = json.load(f)
    # Extract data
    accuracies_1 = train_info_1["accuracies"]
    losses_1 = train_info_1["losses"]
    time_1 = train_info_1["time"]

    accuracies_2 = train_info_2["accuracies"]
    losses_2 = train_info_2["losses"]
    time_2 = train_info_2["time"]
    print('accuracie', Average(accuracies_1), Average(accuracies_2))
    print('losses', Average(losses_1), Average(losses_2))
    print('time', Average(time_1), Average(time_2))
    print(f'time optimize {round((float(Average(time_1)) - float(Average(time_2))) / float(Average(time_1))*100, 2)}%')
    print
    # Plotting
    plt.figure(figsize=(15, 5))

    # Accuracies
    plt.subplot(1, 3, 1)
    plt.plot(accuracies_1, label="stock python")
    plt.plot(accuracies_2, label="intel python")
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Comparison")
    plt.legend()

    # Losses
    plt.subplot(1, 3, 2)
    plt.plot(losses_1, label="stock python")
    plt.plot(losses_2, label="intel python")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Loss Comparison")
    plt.legend()

    # Time
    plt.subplot(1, 3, 3)
    plt.plot(time_1, label="stock python")
    plt.plot(time_2, label="intel python")
    plt.xlabel("Iteration")
    plt.ylabel("Time (s/items)")
    plt.title("Time Comparison")
    plt.legend()

    plt.tight_layout()
    # Save the figure
    plt.savefig(output_filename)

# Function to find the largest folder name in the "result" directory
def find_largest_folder():
    result_dir = "result"
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    os.chdir(result_dir)
    folders = [folder for folder in os.listdir() if os.path.isdir(folder) and folder.isdigit()]
    if folders:
        return max(map(int, folders))
    else:
        return 0

# Function to create a new folder with a name one greater than the largest existing folder
def create_next_folder():
    largest_folder = find_largest_folder()
    new_folder_name = str(largest_folder + 1)
    os.mkdir(new_folder_name)
    print(f"Created folder: {new_folder_name}")
    return new_folder_name

# Function to move all data from "data/federat" to the newly created folder
def move_data_to_new_folder(new_folder_name):
    data_source = "data/federat"
    for item in os.listdir(data_source):
        source = os.path.join(data_source, item)
        if os.path.isfile(source):
            shutil.move(source, os.path.join(new_folder_name, item))
        elif os.path.isdir(source):
            shutil.move(source, os.path.join(new_folder_name, item))

if __name__ == "__main__":
    # generate_comparison_graphs("data/init/", "init_comparison_graphs.png")
    generate_comparison_graphs("data/federat/", "data/federat/comparison_graphs.png")
    new_folder_name = create_next_folder()
    move_data_to_new_folder(new_folder_name)