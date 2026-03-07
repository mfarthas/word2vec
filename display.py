from rich.console import Console
from rich.progress import (
    Progress, SpinnerColumn, BarColumn,
    TextColumn, TimeElapsedColumn, TaskProgressColumn
)
from rich.table import Table
from rich.panel import Panel
from rich import box

console = Console()


def training_header(epochs, embed_dim, window_size, k, lr_start, lr_min):
    console.print(Panel.fit(
        f"[bold cyan]epochs[/] {epochs}  "
        f"[bold cyan]embed_dim[/] {embed_dim}  "
        f"[bold cyan]window[/] {window_size}  "
        f"[bold cyan]k[/] {k}  "
        f"[bold cyan]lr[/] {lr_start} -> {lr_min}",
        title="[bold white]Word2Vec SGNS — Training",
        border_style="cyan"
    ))


def corpus_loaded(n_tokens, vocab_size):
    console.print("\n[bold white][1/4][/] Loading corpus...")
    console.print(
        f"  [green]tokens:[/] {n_tokens:,}  "
        f"[green]vocab:[/] {vocab_size:,}"
    )


def subsampling_done(kept_pct):
    console.print("[bold white][2/4][/] Subsampling...")
    console.print(f"  [green]kept:[/] {kept_pct:.1f}%")


def noise_done():
    console.print("[bold white][3/4][/] Building noise distribution...")
    console.print("  [green]done[/]")


def embeddings_init(shape, mb):
    console.print("[bold white][4/4][/] Initializing embeddings...")
    console.print(
        f"  [green]shape:[/] {shape}  "
        f"[green]size:[/] {mb:.1f} MB each"
    )


def epoch_header(epoch, epochs):
    console.print(f"\n[bold yellow]Epoch {epoch}/{epochs}[/]")


def make_progress(lr_start):
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=35),
        TaskProgressColumn(),
        TextColumn("[green]{task.fields[loss]:.4f}[/] loss"),
        TextColumn("[cyan]{task.fields[lr]:.5f}[/] lr"),
        TextColumn("[white]{task.fields[speed]:.0f}[/] steps/s"),
        TimeElapsedColumn(),
        console=console,
    )


def epoch_done(avg_loss, elapsed):
    console.print(
        f"  [bold green]done[/]  "
        f"avg loss [bold]{avg_loss:.4f}[/]  "
        f"time [bold]{elapsed:.1f}s[/]"
    )


def saved():
    console.print("\n[bold green]✓[/] Embeddings saved to disk")


def neighbors_table(results_per_word):
    console.print(Panel.fit(
        "[bold white]Nearest Neighbors[/]",
        border_style="cyan"
    ))
    table = Table(box=box.SIMPLE, show_header=True)
    table.add_column("word", style="bold cyan", width=10)
    for col in ["1st", "2nd", "3rd", "4th", "5th"]:
        table.add_column(col, width=16)
    for word, results in results_per_word:
        cells = [f"{w} [dim]{s:.3f}[/]" for w, s in results]
        table.add_row(word, *cells)
    console.print(table)


def analogies_table(results_per_query):
    console.print(Panel.fit(
        "[bold white]Word Analogies[/]",
        border_style="cyan"
    ))
    for a, b, c, results in results_per_query:
        answers = "  ".join(
            f"[bold green]{w}[/] [dim]{s:.3f}[/]"
            for w, s in results
        )
        console.print(
            f"  [cyan]{a}[/]:[cyan]{b}[/] :: "
            f"[cyan]{c}[/]:? [white]->[/]  {answers}"
        )
