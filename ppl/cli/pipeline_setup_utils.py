"""Utility functions for the command-line interface setup."""
import argparse
import logging
from pathlib import Path

# Exit codes following POSIX conventions
SUCCESS_EXIT_CODE = 0
ERROR_EXIT_CODE = 1
SIGINT_EXIT_CODE = 130  # POSIX SIGINT (128+2)

# Pipeline configuration
DEFAULT_SEED = 42  # Fixed seed for reproducibility

def get_default_config_path() -> Path:
    """Get the default configuration path."""
    try:
        # ppl/cli/pipeline_setup_utils.py -> ppl/
        return Path(__file__).parent.parent / "config/experiment_configs/run_config.yaml"
    except NameError:
        # Fallback if __file__ is not available
        return Path.cwd() / "ppl/config/experiment_configs/run_config.yaml"

DEFAULT_CONFIG_PATH = get_default_config_path()

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for pipeline configuration."""
    try:
        parser = argparse.ArgumentParser(
            description=(
                "Run a Multi-Instance Learning (MIL) pipeline.\n"
                "Supports custom YAML configuration files with optional logging verbosity control."
            ),
            epilog=(
                "Example usage:\n"
                "  $ poetry run milk                            # uses default config and INFO log-level\n"
                "  $ poetry run milk -c path/to/config.yaml     # run with custom config\n"
                "  $ poetry run milk --log-level DEBUG          # enable debug-level logging\n"
            ),
            formatter_class=argparse.RawTextHelpFormatter,
        )

        parser.add_argument(
            "-c", "--config",
            dest="ppl_cfg_path",
            type=Path,
            help=(
                "Path to the YAML configuration file.\n"
                f"Defaults to '{DEFAULT_CONFIG_PATH.relative_to(Path.cwd())}'."
            ),
        )

        parser.add_argument(
            "--log-level",
            default="INFO",
            choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            help=(
                "Set the verbosity level of log output.\n"
                "Default: INFO"
            ),
        )
        
        args = parser.parse_args(argv)

        # Set default config path if not specified
        if not args.ppl_cfg_path:
            args.ppl_cfg_path = DEFAULT_CONFIG_PATH

        return args
    except Exception as e:
        logging.error("Failed to parse arguments: %s", e)
        raise

# Logger names whose INFO messages form the clean, human-readable run narrative.
# Everything else under ``ppl.*`` is treated as technical detail and hidden at
# INFO (still shown at --log-level DEBUG).
_NARRATIVE_LOGGERS = ("milk", "root", "ppl.models.lightning.training")


class _CleanFormatter(logging.Formatter):
    """Bare message for INFO; ``LEVEL: message`` for warnings/errors."""

    def format(self, record: logging.LogRecord) -> str:
        if record.levelno <= logging.INFO:
            return record.getMessage()
        return f"{record.levelname}: {record.getMessage()}"


class _NarrativeFilter(logging.Filter):
    """At INFO, pass only the run narrative + warnings; hide technical chatter."""

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.WARNING:
            return True
        name = record.name
        return any(name == n or name.startswith(n + ".") for n in _NARRATIVE_LOGGERS)


def configure_logging(level: str = "INFO") -> None:
    """Configure logging.

    At the default ``INFO`` level the output is a minimal, staged narrative
    (the ``milk`` logger plus per-epoch metrics); all the technical ``ppl.*``
    detail is suppressed. Use ``--log-level DEBUG`` to see everything.
    """
    try:
        root_logger = logging.getLogger()
        root_logger.setLevel(level)
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        handler = logging.StreamHandler()
        if str(level).upper() == "DEBUG":
            handler.setFormatter(logging.Formatter(
                "[{asctime}] {levelname:<8} {name}: {message}",
                style="{", datefmt="%Y-%m-%d %H:%M:%S",
            ))
        else:
            handler.setFormatter(_CleanFormatter())
            handler.addFilter(_NarrativeFilter())
        root_logger.addHandler(handler)

        logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
        logging.getLogger("tqdm").setLevel(logging.WARNING)
        # MLflow uses its own root handler; quiet its info chatter.
        if str(level).upper() != "DEBUG":
            logging.getLogger("mlflow").setLevel(logging.WARNING)

    except Exception as e:
        print(f"Failed to configure logging: {e}")
        raise
