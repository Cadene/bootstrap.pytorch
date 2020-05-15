from pathlib import Path
from argparse import ArgumentParser


def replace_content(file_path, prj_name):
    content = file_path.read_text()
    content = content.replace('{PROJECT_NAME}', prj_name)
    content = content.replace('{PROJECT_NAME_LOWER}', prj_name.lower())
    content = content.replace('  # noqa: E999', '')
    return content


def new_project(prj_name, prj_dir):
    # will be rename into project_name.lower() + suffix
    # ex: dataset.py -> myproject.py
    files_to_rename = ['dataset.py', 'criterion.py', 'metric.py', 'network.py', 'options.yaml']

    # will be rename into project_name.lower()
    # ex: project/datasets -> myproject/datasets
    dirs_to_rename = ['project']

    path = Path(prj_dir)
    path = path / f'{prj_name.lower()}.bootstrap.pytorch'
    path.mkdir()
    tpl_path = Path(__file__).parent / 'templates' / 'default'

    print(f'Creating project {prj_name.lower()} in {path}')

    # recursive iteration over directories and files
    for p in tpl_path.rglob('*'):

        # absolute path to local path
        # ex: bootstrap.pytorch/templates/default/project -> project
        tpl_local_path = p.relative_to(tpl_path)

        # replace name of directories
        local_path = p.relative_to(tpl_path)
        for dir_name in dirs_to_rename:
            local_path = Path(str(local_path).replace(dir_name, prj_name.lower()))

        if p.is_dir():
            Path(path / local_path).mkdir()

        if p.is_file():
            content = replace_content(tpl_path / tpl_local_path, prj_name)
            if p.name in files_to_rename:
                local_path = Path(local_path.parent / f'{prj_name.lower()}{p.suffix}')
            print(local_path)
            Path(path / local_path).write_text(content)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--project_name', type=str, default='MyProject')
    parser.add_argument('--project_dir', type=str, default='.')
    args = parser.parse_args()
    new_project(args.project_name, args.project_dir)
