import platform
import sys
import subprocess
from pathlib import Path
import argparse


def make_parser():
    parse = argparse.ArgumentParser('Installer')
    parse.add_argument('-v', '--version',
                       default='cpu',
                       help='Backend version of PyTorch (cpu or cuda)')


    return parse


def install_packages(device):

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
    
    cmd_torch = [sys.executable, '-m', 'pip',
                 'install',
                 '-r',
                 str(packages/install_file_torch.format(device))]

    subprocess.run(cmd_torch, check=True)

    cmd_t_geom =  [i for i in[sys.executable, '-m',
                              'pip',
                              'install',
                              '-r',
                              str(packages/install_file_t_geom.format(device)),
                              add_args] if i]

    subprocess.run(cmd_t_geom, check=True)

    print('All required PyTorch packages installed successfully')




def install_project():
    cmd = [sys.executable,
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

    device = args.version.strip().lower()

    if device not in ['cpu', 'cuda']:
        print('Wrong PyTorch backend version')
        print('cpu or cuda currently available')
        return


    install_packages(device)
    install_project()


if __name__ == '__main__':
    main()
