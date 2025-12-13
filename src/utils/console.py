"""
Console output utilities for experiments.

Provides colored output and formatted printing.
"""

import sys
from dataclasses import asdict


# =============================================================================
# COLORS
# =============================================================================

class Colors:
    """ANSI color codes."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    RESET = '\033[0m'


# Disable colors if not TTY
if not sys.stdout.isatty():
    for attr in ['HEADER', 'BLUE', 'CYAN', 'GREEN', 'YELLOW', 'RED', 'BOLD', 'DIM', 'RESET']:
        setattr(Colors, attr, '')


# =============================================================================
# PRINT FUNCTIONS
# =============================================================================

def print_header(title: str, experiment_num: int = None):
    """Print experiment header."""
    line = "=" * 60
    print(f"\n{Colors.BOLD}{Colors.CYAN}{line}{Colors.RESET}")
    if experiment_num:
        print(f"{Colors.BOLD}{Colors.CYAN}  EXPERIMENT {experiment_num:02d}: {title}{Colors.RESET}")
    else:
        print(f"{Colors.BOLD}{Colors.CYAN}  {title}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{line}{Colors.RESET}\n")


def print_section(title: str):
    """Print section header."""
    print(f"\n{Colors.BOLD}[{title}]{Colors.RESET}")
    print("-" * 40)


def print_config(config):
    """Print configuration in a readable format."""
    if hasattr(config, '__dataclass_fields__'):
        for key, value in asdict(config).items():
            print(f"  {key}: {value}")
    else:
        for key, value in vars(config).items():
            if not key.startswith('_'):
                print(f"  {key}: {value}")


def print_results_table(results: dict, x_values: list, title: str = "Results"):
    """Print results comparison table."""
    print(f"\n{Colors.BOLD}=== {title.upper()} ==={Colors.RESET}")

    # Header
    header = f"{'Labels':<10}"
    for name in results.keys():
        header += f" | {name:<12}"
    print(header)
    print("-" * len(header))

    # Rows
    for i, x in enumerate(x_values):
        row = f"{x:<10}"
        for name, values in results.items():
            if i < len(values):
                row += f" | {values[i]:>10.2f}%"
            else:
                row += f" | {'N/A':>10}"
        print(row)


def print_success(message: str):
    """Print success message."""
    print(f"{Colors.GREEN}✓ {message}{Colors.RESET}")


def print_error(message: str):
    """Print error message."""
    print(f"{Colors.RED}✗ {message}{Colors.RESET}")


def print_warning(message: str):
    """Print warning message."""
    print(f"{Colors.YELLOW}⚠ {message}{Colors.RESET}")


def print_info(message: str):
    """Print info message."""
    print(f"{Colors.BLUE}ℹ {message}{Colors.RESET}")

