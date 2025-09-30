import fiftyone as fo

# List available datasets
datasets = fo.list_datasets()

if not datasets:
    print("❌ No datasets found in FiftyOne.")
    exit()

print("\n📂 Available Datasets:")
for idx, name in enumerate(datasets):
    print(f"{idx}: {name}")

# Prompt user to select a dataset
while True:
    try:
        choice = int(input("\nEnter the index of the dataset you want to load: "))
        if 0 <= choice < len(datasets):
            selected_dataset = datasets[choice]
            break
        else:
            print("Invalid index. Please try again.")
    except ValueError:
        print("Invalid input. Enter a number.")

# Prompt user for custom port (default 5151)
try:
    port_input = input("Enter the port number to launch FiftyOne (default = 5151): ").strip()
    port = int(port_input) if port_input else 5151
except ValueError:
    print("Invalid port. Using default port 5151.")
    port = 5151

print(f"\n✅ Loading dataset: '{selected_dataset}' on port {port}")
dataset = fo.load_dataset(selected_dataset)

# Make dataset persistent
dataset.persistent = True
dataset.save()

# Launch FiftyOne app on specified port
session = fo.launch_app(dataset, port=port)
session.wait()

# Example SSH command for remote access (optional reminder)
# ssh -L 5151:localhost:5151 user@remote_host
