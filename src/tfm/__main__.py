"""Permite `python -m tfm run all` como alternativa al comando `tfm`."""

import sys

from tfm.pipeline.cli import main

sys.exit(main())
