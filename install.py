import platform
import sys
import subprocess
from pathlib import Path
import argparse


def make_parser():
    parse = argparse.ArgumentParser('Installer')
    parse.add_argument(
        '-d', '--device',
        default='cpu',
        help='Backend version of PyTorch (cpu or cuda)'
    )

    parse.add_argument(
        '-f', '--file',
        default='environment.yaml',
        help='Path to Conda Environment File'
    )


    return parse


def conda_env_name_extract(
        conda_env_file: str = 'environment.yaml'
) -> str:
    conda_env_file = 'environment.yaml'
    args = [
        'grep',
        'name',
        conda_env_file
    ]

    result = subprocess.run(
        args=args, 
        capture_output=True,
        text=True,
        encoding='utf-8')

    return result.stdout.split()[1]


def build_conda_enviroment(
        conda_env_file: str='environment.yaml'
):

    args = [
        'conda',
        'create',
        '-f',
        conda_env_file,
        '-y'
    ]

    subprocess.run(args)


def install_packages(
        device: str,
        conda_env_name: str
):

    # Packages directory
    packages = Path('packages')

    
    install_file_torch = 'requirements_torch_{}.txt'
    install_file_t_geom = 'requirements_geom_{}.txt'
    add_args = None
    
    if platform.system() == "Darwin":
    
        print('MacOS system detected')
        print('WARNING: PyTorch will be installed with cpu backend and --no-build-isolation')
    
        device = 'cpu'
        add_args = '--no-build-isolation'
    
    cmd_torch = [
        'conda',
        'run',
        '-n',
        conda_env_name,
        'python',
        '-m', 
        'pip',
        'install',
        '-r',
        str(packages/install_file_torch.format(device))]

    subprocess.run(cmd_torch, check=True)

    cmd_t_geom =  [i for i in ['conda',
                               'run',
                               '-n',
                               conda_env_name,
                               'python', 
                               '-m',
                               'pip',
                               'install',
                               '-r',
                               str(packages/install_file_t_geom.format(device)),
                               add_args] if i]

    subprocess.run(cmd_t_geom, check=True)
    print('All required PyTorch packages installed successfully')


def install_project(
        conda_env_name: str
):
    cmd = [
        'conda',
        'run',
        '-n',
        conda_env_name,
        'python',
        '-m',
        'pip',
        'install',
        '-e',
        '.']

    subprocess.run(cmd, check=True)
    print('Project GNN Affinity installed successfully')
    print('To initialize it, activate the conda environment: gnn-aff')
    print('Run the program with: gnn-affinity')
    print('For help, type: gnn-affinity -h')

    



def main():

    args = make_parser()
    args = args.parse_args()

    device = args.device.strip().lower()
    conda_env_file = args.file

    if device not in ['cpu', 'cuda']:
        print('Wrong PyTorch backend version')
        print('cpu or cuda currently available')
        return

    build_conda_enviroment(conda_env_file)
    conda_env_name = conda_env_name_extract(conda_env_file)

    install_packages(device, conda_env_name)
    install_project(conda_env_name)



if __name__ == '__main__':
    main()