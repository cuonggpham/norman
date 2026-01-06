#!/usr/bin/env python3
"""
CLI Testing Tool for Norman API

Usage:
    python scripts/cli.py health
    python scripts/cli.py search "労働時間" --top-k 5
    python scripts/cli.py chat "Quy định về thời gian làm việc" --top-k 5
"""

import json
from typing import Optional

import httpx
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich.json import JSON

app = typer.Typer(
    name="norman-cli",
    help="CLI tool for testing Norman RAG API endpoints"
)
console = Console()

BASE_URL = "http://localhost:8000/api"


@app.command()
def health():
    """Check API health status."""
    try:
        response = httpx.get(f"{BASE_URL}/health", timeout=10)
        response.raise_for_status()
        data = response.json()
        
        console.print(Panel(
            f"[green]Status: {data['status']}[/green]\n"
            f"Version: {data['version']}\n"
            f"Services: {json.dumps(data['services'], indent=2)}",
            title="🏥 Health Check",
            border_style="green"
        ))
    except httpx.RequestError as e:
        console.print(f"[red]❌ Connection error: {e}[/red]")
        raise typer.Exit(1)
    except httpx.HTTPStatusError as e:
        console.print(f"[red]❌ HTTP error: {e.response.status_code}[/red]")
        raise typer.Exit(1)


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of results"),
    filters: Optional[str] = typer.Option(None, "--filters", "-f", help="JSON filters"),
    raw: bool = typer.Option(False, "--raw", "-r", help="Output raw JSON"),
):
    """Vector search for legal documents."""
    try:
        payload = {"query": query, "top_k": top_k}
        if filters:
            payload["filters"] = json.loads(filters)
        
        console.print(f"[dim]🔍 Searching: {query}[/dim]")
        
        response = httpx.post(
            f"{BASE_URL}/search",
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
        
        if raw:
            console.print(JSON(json.dumps(data, ensure_ascii=False, indent=2)))
            return
        
        # Rich formatted output
        console.print(Panel(
            f"Query: [cyan]{data['query']}[/cyan]\n"
            f"Total: [yellow]{data['total']}[/yellow] results\n"
            f"Time: [dim]{data['processing_time_ms']:.1f}ms[/dim]",
            title="🔍 Search Results",
            border_style="blue"
        ))
        
        for i, result in enumerate(data['results'], 1):
            table = Table(show_header=False, box=None, padding=(0, 1))
            table.add_column("Key", style="dim")
            table.add_column("Value")
            
            table.add_row("Law", f"[bold]{result['law_title']}[/bold]")
            table.add_row("Article", result['article_title'])
            if result.get('chapter_title'):
                table.add_row("Chapter", result['chapter_title'])
            table.add_row("Score", f"[green]{result['score']:.4f}[/green]")
            
            console.print(Panel(
                table,
                title=f"#{i}",
                subtitle=f"[dim]{result['chunk_id'][:20]}...[/dim]",
                border_style="cyan"
            ))
            console.print(f"  [dim]{result['text'][:300]}...[/dim]\n")
            
    except httpx.RequestError as e:
        console.print(f"[red]❌ Connection error: {e}[/red]")
        raise typer.Exit(1)
    except httpx.HTTPStatusError as e:
        console.print(f"[red]❌ HTTP error: {e.response.status_code}[/red]")
        console.print(f"[dim]{e.response.text}[/dim]")
        raise typer.Exit(1)
    except json.JSONDecodeError:
        console.print("[red]❌ Invalid JSON in filters[/red]")
        raise typer.Exit(1)


@app.command()
def chat(
    query: str = typer.Argument(..., help="Question to ask"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of sources"),
    filters: Optional[str] = typer.Option(None, "--filters", "-f", help="JSON filters"),
    sources: bool = typer.Option(True, "--sources/--no-sources", "-s", help="Show sources"),
    raw: bool = typer.Option(False, "--raw", "-r", help="Output raw JSON"),
):
    """RAG chat - get answers with sources."""
    try:
        payload = {"query": query, "top_k": top_k}
        if filters:
            payload["filters"] = json.loads(filters)
        
        console.print(f"[dim]💬 Asking: {query}[/dim]")
        console.print(f"[dim]⏳ Generating response...[/dim]\n")
        
        response = httpx.post(
            f"{BASE_URL}/chat",
            json=payload,
            timeout=60  # LLM can take time
        )
        response.raise_for_status()
        data = response.json()
        
        if raw:
            console.print(JSON(json.dumps(data, ensure_ascii=False, indent=2)))
            return
        
        # Answer panel with markdown
        console.print(Panel(
            Markdown(data['answer']),
            title="💬 Answer",
            border_style="green",
            padding=(1, 2)
        ))
        
        console.print(f"[dim]⏱️  Processing time: {data['processing_time_ms']:.1f}ms[/dim]\n")
        
        # Sources table
        if sources and data.get('sources'):
            console.print("[bold]📚 Sources:[/bold]")
            
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("#", style="dim", width=3)
            table.add_column("Law", style="cyan")
            table.add_column("Article", style="green")
            table.add_column("Score", justify="right")
            
            for i, src in enumerate(data['sources'], 1):
                table.add_row(
                    str(i),
                    src['law_title'],
                    src['article'],
                    f"{src['score']:.3f}"
                )
            
            console.print(table)
            
    except httpx.RequestError as e:
        console.print(f"[red]❌ Connection error: {e}[/red]")
        raise typer.Exit(1)
    except httpx.HTTPStatusError as e:
        console.print(f"[red]❌ HTTP error: {e.response.status_code}[/red]")
        console.print(f"[dim]{e.response.text}[/dim]")
        raise typer.Exit(1)
    except json.JSONDecodeError:
        console.print("[red]❌ Invalid JSON in filters[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
