def get_git_commit(repo_path: str = ".") -> dict:
    """Get git commit information including hash, author, and message.

    Args:
        repo_path: Path to the git repository (default: current directory)

    Returns:
        Dictionary containing 'hash', 'author', and 'message'
    """
    import subprocess

    try:
        # Get commit hash
        commit_hash = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repo_path, stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )

        # Get commit author
        author = (
            subprocess.check_output(
                ["git", "log", "-1", "--pretty=format:%an <%ae>"],
                cwd=repo_path,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )

        # Get commit message
        message = (
            subprocess.check_output(
                ["git", "log", "-1", "--pretty=format:%s"],
                cwd=repo_path,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )

        return {"hash": commit_hash, "author": author, "message": message}
    except subprocess.CalledProcessError:
        return {"hash": "unknown", "author": "unknown", "message": "unknown"}


def read_file(path: str) -> str:
    """Read a text file and return its content as a string."""
    with open(path, "r") as f:
        return f.read()


def write_csv(df, path: str) -> None:
    """Write a DataFrame to CSV."""
    df.to_csv(path, index=False)


def write_json(metadata, df_kernels, df_regions, path: str) -> None:
    """Write DataFrames to JSON."""
    import json

    output = {
        "metadata": metadata,
        "kernels": df_kernels.to_dict(orient="records"),
        "regions": df_regions.to_dict(orient="records"),
    }
    with open(path, "w") as f:
        json.dump(output, f, indent=4)
