import yaml


def read_yaml(file_path):
    """
    Reads a YAML file and returns its contents as a dictionary.

    Arguments:
        file_path: str, path to the YAML file to read.
    Returns:
        dict: contents of the YAML file.
    """
    with open(file_path, "r") as file:
        try:
            data = yaml.safe_load(file)
            return data
        except yaml.YAMLError as exc:
            print(f"Error parsing YAML: {exc}")
