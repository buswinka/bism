import argparse
from bism.eval.run import run_model


def main():
    """ Entry point for all bism evaluation scripts. See bism.eval.run.run_model() for more detail """
    parser = argparse.ArgumentParser(description="BISM EVAL Parameters")
    parser.add_argument("-m", "--model_file", type=str, help="YAML config file for training")
    parser.add_argument("-i", "--image_path", type=str, help="Path to image")
    parser.add_argument('-d', '--device', type=str, help='hardware accelerator', default='Infer')
    parser.add_argument(
        "--log",
        type=int,
        default=3,
        choices=[0,1,2,3,4],
        help="Log Level: 0-Debug, 1-Info, 2-Warning, 3-Error, 4-Critical",
    )

    args = parser.parse_args()
    run_model(model_file=args.model_file, image_path=args.image_path, device=args.device, log_level=args.log)


if __name__ == "__main__":
    import sys

    sys.exit(main())
