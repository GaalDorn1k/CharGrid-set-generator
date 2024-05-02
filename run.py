import yaml
import argparse

from plotter import Plotter
from generator import Generator


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config.yaml', help='Config file')
    parser.add_argument('-d', '--debug', type=bool, default=True, help='Save debug images')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    generator = Generator(config)
    generator.generate()

    if args.debug:
        plotter = Plotter('gen_data')
        plotter.plot(plot_rows=True, plot_chars=True)
