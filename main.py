from cli.cli_args import make_parser


from rdkit import RDLogger

RDLogger.DisableLog("rdApp.error")
RDLogger.DisableLog("rdApp.warning")


def main():
    parser = make_parser()
    args = parser.parse_args()
    args.func(args)



if __name__ == '__main__':
    main()