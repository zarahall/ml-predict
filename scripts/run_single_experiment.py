#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

"""
Run an experiment
"""

import yaml

from preference_inferrer.tasks import run_task
import prose_vocabulary_constraint

RUN_NOTES = "validate using costs"


def main():
    """Main"""

    with open("experiment_config.yaml", 'r') as file:
        config = yaml.safe_load(file)["parameters"]
    run_task.run(config)


if __name__ == '__main__':
    main()
