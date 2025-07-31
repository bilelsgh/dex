import sys

import pytest


def run_tests():
    """Lance tous les tests avec un rapport détaillé"""

    # Options de pytest
    pytest_args = [
        "-v",
        "--tb=short",
        "--color=yes",
        "--durations=10",
        "--cov=your_module",
        "--cov-report=html",
    ]

    # Lancer les tests
    exit_code = pytest.main(pytest_args)

    if exit_code == 0:
        print("✅ Tous les tests sont passés avec succès!")
    else:
        print("❌ Certains tests ont échoué.")

    return exit_code


if __name__ == "__main__":
    sys.exit(run_tests())
