""" Command-line entry point for Multi-Instance Learning Kit (MILK). Built on PyTorch Lightning. """
import logging
import sys

from ppl.utils.cli import (
    execute_pipeline,
    parse_args,
    ERROR_EXIT_CODE,
    SIGINT_EXIT_CODE,
    SUCCESS_EXIT_CODE,
)

def _map_exception_to_exit_code(exc: Exception) -> int:
    """Map exceptions to appropriate exit codes with corresponding log messages.

    Args:
        exc: The exception that was raised

    Returns:
        The appropriate exit code for the given exception
    """
    if isinstance(exc, KeyboardInterrupt):
        logging.warning("Interrupted by user.")
        return SIGINT_EXIT_CODE
    elif isinstance(exc, FileNotFoundError):
        logging.error("Configuration file error: %s", exc)
        return ERROR_EXIT_CODE
    else:
        logging.exception("Pipeline failed.")
        return ERROR_EXIT_CODE

def run(argv: list[str] | None = None) -> int:
    """Execute the MILK pipeline with the provided arguments.

    Args:
        argv: Command line arguments (uses sys.argv if None)

    Returns:
        Exit code indicating success (0) or failure (non-zero)
    """
    try:
        args = parse_args(argv)
        execute_pipeline(args)
        return SUCCESS_EXIT_CODE
    except Exception as e:
        return _map_exception_to_exit_code(e)

def main() -> None:
    """Command-line entry point for the MILK pipeline."""
    sys.exit(run())


if __name__ == "__main__":
    main()
