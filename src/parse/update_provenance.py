import json
import logging
from pathlib import Path
from datetime import datetime


def update_parser_provenance(company_ticker: str, parser_details: list, parsed_files: list = None, parsed_files_summary: str = None):
    """
    Update parser_provenance.json with parsing details provenance info and parsed files list.
    Args:
        company_ticker (str): The company ticker (e.g., 'NVDA')
        parser_details (list): List of parsing detail dicts (with parsing stats)
        parsed_files (list): List of all parsed file paths
        parsed_files_summary (str): Summary of parsed files
    """
    metadata_dir = Path(f"data/metadata/{company_ticker}")
    metadata_dir.mkdir(parents=True, exist_ok=True)
    provenance_file = metadata_dir / "parser_provenance.json"

    # Load existing provenance if present
    if provenance_file.exists():
        with open(provenance_file, "r", encoding="utf-8") as f:
            provenance = json.load(f)
    else:
        provenance = {}

    # Add/update parser provenance
    provenance_entry = {
        "updated": datetime.now().isoformat(),
        "parsing_count": len(parser_details),
        "parsing_details": parser_details
    }
    if parsed_files is not None:
        provenance_entry["parsed_files"] = parsed_files
    if parsed_files_summary is not None:
        provenance_entry["parsed_files_summary"] = parsed_files_summary

    provenance["parser_provenance"] = provenance_entry

    with open(provenance_file, "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2, ensure_ascii=False)
    logging.info(f"Updated parser provenance for {company_ticker}: {provenance_file}")

# Example usage:
# pdf_links = [{"url": "https://example.com/report.pdf", "title": "Q4 Report"}]
# update_discover_provenance("NVDA", pdf_links)
