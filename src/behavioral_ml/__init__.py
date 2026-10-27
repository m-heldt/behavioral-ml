def main() -> None:
    """Console entry point; defer ML imports until invoked."""
    from .experiment import main as run
    run()
