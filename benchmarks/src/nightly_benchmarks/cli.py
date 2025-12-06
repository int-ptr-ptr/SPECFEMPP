import argparse
from .kp_parser import (
    parse_kokkos_output,
    parse_regions_section,
    parse_metadata,
    total_execution_time,
)
from .io_utils import read_file, write_json, get_git_commit


def main():
    parser = argparse.ArgumentParser(description="Parse Kokkos profiler output.")
    parser.add_argument("profile", help="Path to kokkos profiler output file")
    parser.add_argument(
        "-o", "--output", default="output.json", help="Path to save parsed output JSON"
    )

    args = parser.parse_args()

    text = read_file(args.profile)
    df_kernels = parse_kokkos_output(text)
    df_regions = parse_regions_section(text)
    metadata = parse_metadata(text)
    total_time = total_execution_time(text)
    ## append total execution time to metadata
    metadata["total_execution_time"] = total_time
    ## append git commit information to metadata
    git_commit = get_git_commit()
    metadata["git_commit"] = git_commit
    # Save outputs
    write_json(metadata, df_kernels, df_regions, args.output)

    print(f"Parsed output saved to {args.output}")
    print(f"Total execution time: {total_time} seconds")
    return 0
