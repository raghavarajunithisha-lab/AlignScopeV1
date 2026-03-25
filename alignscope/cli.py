"""
AlignScope — CLI

Commands:
    alignscope start [--port 8000] [--demo]   Start the dashboard server
    alignscope patch <framework>               Auto-patch a MARL framework
    alignscope share                          Create a public tunnel (ngrok)
    alignscope version                        Print version
"""

import click
from rich.console import Console

console = Console()


@click.group()
def main():
    """🔬 AlignScope — MARL Alignment Observability"""
    pass


@main.command()
@click.option("--port", default=8000, help="Port to serve on")
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--demo", is_flag=True, help="Run built-in demo simulator")
def start(port: int, host: str, demo: bool):
    """Start the AlignScope dashboard server."""
    from alignscope.server import run_server

    console.print()
    console.print("[bold cyan]🔬 AlignScope[/] starting...", highlight=False)
    console.print()

    run_server(host=host, port=port, demo=demo)


@main.command()
@click.argument("framework", type=click.Choice(["rllib", "pettingzoo", "pymarl"]))
def patch(framework: str):
    """Auto-patch a MARL framework for zero-code integration."""
    from alignscope.patches import apply_patch

    console.print(f"\n[bold cyan]🔬 AlignScope[/] patching [bold]{framework}[/]...\n")

    success = apply_patch(framework)

    if success:
        console.print(f"[green]✓[/] [bold]{framework}[/] successfully patched!")
        console.print(f"  Your training code needs [bold]zero changes[/].")
        console.print(f"  Just run [cyan]alignscope start[/] in another terminal,")
        console.print(f"  then run your training script as normal.\n")
    else:
        console.print(f"[red]✗[/] Failed to patch {framework}.")
        console.print(f"  Is it installed? Try: [cyan]pip install alignscope[{framework}][/]\n")


@main.command()
@click.option("--port", default=8000, help="Local port to tunnel")
def share(port: int):
    """Create a public URL to share your dashboard with teammates."""
    try:
        from pyngrok import ngrok

        console.print(f"\n[bold cyan]🔬 AlignScope[/] creating public tunnel...\n")

        tunnel = ngrok.connect(port, "http")
        public_url = tunnel.public_url

        console.print(f"[green]✓[/] Dashboard is now publicly accessible:")
        console.print(f"  [bold link={public_url}]{public_url}[/]")
        console.print(f"\n  Share this URL with your team.")
        console.print(f"  Press [bold]Ctrl+C[/] to stop sharing.\n")

        try:
            ngrok.get_tunnels()
            input("  Press Enter to stop sharing...")
        except KeyboardInterrupt:
            pass
        finally:
            ngrok.disconnect(tunnel.public_url)
            console.print("\n[yellow]Tunnel closed.[/]\n")

    except ImportError:
        console.print(
            "\n[red]✗[/] ngrok is not installed.\n"
            "  Install it with: [cyan]pip install pyngrok[/]\n"
        )


@main.command()
def version():
    """Print AlignScope version."""
    from alignscope import __version__
    console.print(f"AlignScope v{__version__}")


if __name__ == "__main__":
    main()
