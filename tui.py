from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from logic import create_rag_chain


def run_tui():
    """Runs the main Terminal User Interface loop."""
    console = Console()

    # --- Header ---
    header = Text("El Paso Municipal Code Assistant", style="bold blue")
    console.print(Panel(header, title="Welcome!", border_style="green"))
    console.print(
        "Ask a question about city regulations, or type 'exit' to quit.", style="dim")

    # --- Initialize RAG Chain ---
    try:
        with console.status("[bold green]Initializing AI assistant...[/bold green]", spinner="dots"):
            rag_chain = create_rag_chain()
        console.print("[bold green]✅ Assistant is ready.[/bold green]\n")
    except Exception as e:
        console.print(
            f"[bold red]❌ Failed to initialize assistant: {e}[/bold red]")
        return

    # --- Interactive Loop ---
    while True:
        try:
            question = console.input(
                "[bold yellow]Your question: [/bold yellow]")
            if question.lower() == 'exit':
                break

            with console.status("[bold cyan]Searching database and thinking...[/bold cyan]", spinner="earth"):
                response = rag_chain.invoke(question)

            # --- Display Sources ---
            with console.status("[dim]Formatting sources...[/dim]", spinner="point"):
                console.print("[bold green]--- Sources ---[/bold green]")
                for i, doc in enumerate(response["context"]):
                    section = doc.metadata.get("section", "N/A")
                    content = Text(doc.page_content)
                    source_panel = Panel(
                        content,
                        title=f"[bold yellow]Source {i+1}: Section {section}[/bold yellow]",
                        border_style="yellow",
                        expand=True
                    )
                    console.print(source_panel)

            console.print("")  # Add a blank line for spacing

            # --- Display Answer ---
            answer_text = Text(response["answer"])
            console.print(Panel(
                answer_text, title="[bold blue]Assistant's Answer[/bold blue]", border_style="blue", expand=True))

        except KeyboardInterrupt:
            break
        except Exception as e:
            console.print(f"[bold red]An error occurred: {e}[/bold red]")

    console.print("\n[bold blue]Goodbye![/bold blue]")


if __name__ == "__main__":
    run_tui()
