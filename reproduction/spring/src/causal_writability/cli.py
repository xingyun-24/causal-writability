"""Small entry points around the original experiment implementations."""
import argparse
import runpy
import sys


COMMANDS = {
    'data': 'sshv2.simulation.build_spring_shortcuts_v1',
    'encode': 'causal_writability.commands.encode_latents',
    'train': 'causal_writability.commands.train',
    'generate': 'causal_writability.commands.generate_videos',
    'evaluate': 'causal_writability.commands.evaluate_predictions',
    'edit': 'causal_writability.edit',
    'weights': 'causal_writability.weights',
}


def main():
    if len(sys.argv) > 1 and sys.argv[1] in COMMANDS:
        command = sys.argv[1]
        sys.argv = ['cw ' + command, *sys.argv[2:]]
        runpy.run_module(COMMANDS[command], run_name='__main__')
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=COMMANDS)
    args, remainder = parser.parse_known_args()
    sys.argv = ['cw ' + args.command, *remainder]
    runpy.run_module(COMMANDS[args.command], run_name='__main__')


if __name__ == '__main__':
    main()
