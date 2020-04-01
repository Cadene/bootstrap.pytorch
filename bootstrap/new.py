import os
from pathlib import Path
from argparse import ArgumentParser


file_dir = Path(__file__).parent


parser = ArgumentParser()
parser.add_argument("--project_name", type=str, help="Project name")


def get_template_file(filename, project_name):
    parts = list(filename.parts)
    project_index = parts.index(project_name.lower())
    if parts[-1] not in ["__init__.py", "factory.py"]:
        parts[-1] = "template_" + parts[-1][2:]  # remove "my"
    template_path = "/".join(parts[project_index + 1:])
    template_path = file_dir / Path("template") / template_path

    return template_path


def get_file_content(filename, project_name):
    template = get_template_file(filename, project_name)

    content = Path(template).read_text()
    content = content.replace("{PROJECT_NAME}", project_name)
    content = content.replace("{PROJECT_NAME_LOWER}", project_name.lower())
    content = content.replace("{PROJECT_NAME_UPPER}", project_name.upper())

    return content


def write_files(files, project_name):
    for f in files:
        content = get_file_content(f, project_name)
        f.write_text(content)


def get_files(directory):
    dir_name = directory.stem
    if dir_name == "options":
        return [directory / "abstract.yaml"]

    to_ret = []
    if dir_name != "models":
        to_ret.append(directory / "__init__.py")

    to_ret.append(directory / "factory.py")
    custom_file = f"my{dir_name[:-1]}.py"
    to_ret.append(directory / custom_file)

    return to_ret


if __name__ == "__main__":
    args = parser.parse_args()
    project_name = args.project_name

    path = Path(f"{project_name.lower()}.bootstrap.pytorch")
    path.mkdir()

    print(f"Creating logs directory")
    os.mkdir(path / "logs")

    print(f"Creating project directory and __init__.py file")
    path = Path(f"{project_name.lower()}.bootstrap.pytorch/{project_name.lower()}")
    path.mkdir()
    Path(path / "__init__.py").touch()

    print("Creating models directory and __init__ file")
    Path(path / "models").mkdir()
    Path(path / "models/__init__.py").touch()

    directories = [
        "datasets",
        "models/networks",
        "models/criterions",
        "models/metrics",
    ]

    for directory in directories:
        print(f"Creating {directory} folder and associated files")
        new_dir = path / directory
        if directory != "models":
            new_dir.mkdir()
        files = get_files(new_dir)
        write_files(files, project_name)

    print("Project is ready !")
