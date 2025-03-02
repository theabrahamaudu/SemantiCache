import pytest
import sys
import re
import xml.etree.ElementTree as ET
from semanticache.utils.logger import logger as log_handler

logger = log_handler()

README_FILE = "README.md"
BADGE_TEMPLATE = "![Coverage]\
    (https://img.shields.io/badge/coverage-{percentage}%25-brightgreen)"


def extract_coverage_from_xml(report_dir):
    """Extract the total coverage percentage from coverage.xml."""
    try:
        coverage_file = f"{report_dir}/coverage.xml"
        tree = ET.parse(coverage_file)
        root = tree.getroot()
        coverage = root.attrib.get("line-rate")
        if coverage:
            return str(int(float(coverage) * 100))
    except (FileNotFoundError, ET.ParseError):
        pass
    return None


def update_readme_with_badge(coverage):
    """Update README.md with the latest coverage percentage badge."""
    try:
        with open(README_FILE, "r") as file:
            content = file.read()

        new_badge = BADGE_TEMPLATE.format(percentage=coverage)
        if "![Coverage]" in content:
            content = re.sub(r"!\[Coverage\]\(.*?\)", new_badge, content)
        else:
            content = f"{new_badge}\n\n{content}"

        with open(README_FILE, "w") as file:
            file.write(content)

        logger.info("README.md updated with coverage badge: %s" % coverage)

    except FileNotFoundError:
        logger.exception("README.md not found. Skipping badge update.")


def run_tests(report_dir="coverage_report"):
    """Run tests, generate a coverage report, and update README.md."""
    args = [
        "--cov=semanticache.cache",
        f"--cov-report=xml:{report_dir}/coverage.xml",
        "--cov-report=term",
        "--ignore=tests/test_load.py"
    ] + sys.argv[1:]

    exit_code = pytest.main(args)

    coverage = extract_coverage_from_xml(report_dir)
    if coverage:
        update_readme_with_badge(coverage)

    sys.exit(exit_code)


if __name__ == "__main__":
    run_tests()
