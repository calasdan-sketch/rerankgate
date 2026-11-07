"""``python -m rerankgate``: same as the ``rerankgate`` command (works without PATH setup)."""

import sys

from rerankgate.cli import main

sys.exit(main())
