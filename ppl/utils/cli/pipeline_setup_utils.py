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
        return Path(__file__).parent.parent.parent / "utils/experiment_configs/nos_regression_experiment_config.yaml"
    except NameError:
        # Fallback if __file__ is not available
        return Path.cwd() / "utils/experiment_configs/nos_regression_experiment_config.yaml"

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
                "  $ poetry run milk --list-use-cases           # list available use cases\n"
                "  $ poetry run milk --log-level DEBUG          # enable debug-level logging\n"
            ),
            formatter_class=argparse.RawTextHelpFormatter,
        )
        
        # Create a mutually exclusive group for config file or list-use-cases
        config_group = parser.add_mutually_exclusive_group()
        
        config_group.add_argument(
            "-c", "--config",
            dest="ppl_cfg_path",
            type=Path,
            help=(
                "Path to the YAML configuration file.\n"
                f"Defaults to '{DEFAULT_CONFIG_PATH.relative_to(Path.cwd())}'."
            ),
        )
        
        config_group.add_argument(
            "--list-use-cases",
            action="store_true",
            help="List available use cases and exit.",
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
        
        # Set default config path if not specified and not listing use cases
        if not args.ppl_cfg_path and not args.list_use_cases:
            args.ppl_cfg_path = DEFAULT_CONFIG_PATH
            
        return args
    except Exception as e:
        logging.error("Failed to parse arguments: %s", e)
        raise

def configure_logging(level: str = "INFO") -> None:
    """Configure a root logger with consistent formatting and timestamp.

    This configuration is designed to be compatible with tqdm progress bars
    by using a simpler format during training and ensuring logs don't interfere
    with the progress bar.
    """
    try:
        # Create a filter to suppress certain log messages during training
        class ProgressBarFilter(logging.Filter):
            def filter(self, record):
                # Filter out memory usage logs during training steps
                if "_step" in record.getMessage() and record.levelno < logging.WARNING:
                    return False
                return True

        # Configure the root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(level)

        # Remove any existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Create a new handler with the filter
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "[{asctime}] {levelname:<8} {name}: {message}",
            style="{",
            datefmt="%Y-%m-%d %H:%M:%S"
        ))
        handler.addFilter(ProgressBarFilter())
        root_logger.addHandler(handler)

        # Set PyTorch Lightning's logger level to WARNING to reduce output
        logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

        # Set tqdm logger level to WARNING to avoid interference with progress bars
        logging.getLogger("tqdm").setLevel(logging.WARNING)

    except Exception as e:
        print(f"Failed to configure logging: {e}")
        raise
