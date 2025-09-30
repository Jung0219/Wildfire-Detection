import fiftyone as fo

def delete_datasets_interactively():
    while True:
        datasets = fo.list_datasets()

        if not datasets:
            print("No datasets available.")
            break

        print("\nAvailable Datasets:")
        for idx, name in enumerate(datasets):
            print(f"{idx}: {name}")

        user_input = input("\nEnter index or name of dataset to delete (or type 'exit' to quit): ").strip()

        if user_input.lower() == "exit":
            print("Exiting.")
            break

        # Try index-based selection
        if user_input.isdigit():
            idx = int(user_input)
            if 0 <= idx < len(datasets):
                dataset_name = datasets[idx]
            else:
                print("Invalid index. Try again.")
                continue
        else:
            dataset_name = user_input if user_input in datasets else None
            if dataset_name is None:
                print("Dataset not found. Try again.")
                continue

        confirm = input(f"Are you sure you want to delete '{dataset_name}'? [y/N]: ").strip().lower()
        if confirm == "y":
            fo.delete_dataset(dataset_name)
            print(f"Deleted dataset '{dataset_name}'.")
        else:
            print("Deletion cancelled.")

if __name__ == "__main__":
    delete_datasets_interactively()

"""
lsof -i :5151
if something is (LISTEN), then kill that process
kill 487975
"""