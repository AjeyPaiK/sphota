"""Training loop for sandhi-viccheda model."""

import json
import math
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn
from rich.table import Table
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

console = Console()


class Trainer:
    """Trainer for sandhi-viccheda model."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        checkpoint_dir: str = "checkpoints",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        lr: float = 3e-4,
        warmup_steps: int = 4000,
        accum_steps: int = 4,
        grad_clip: float = 1.0,
        label_smoothing: float = 0.1,
        use_amp: bool = True,
    ):
        """Initialize trainer.

        Args:
            model: Transformer model
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            checkpoint_dir: Directory for saving checkpoints
            device: Device (cuda or cpu)
            lr: Learning rate
            warmup_steps: Number of warmup steps
            accum_steps: Gradient accumulation steps
            grad_clip: Gradient clipping max norm
            label_smoothing: Label smoothing for cross entropy
            use_amp: Use automatic mixed precision (fp16)
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.warmup_steps = warmup_steps
        self.accum_steps = accum_steps
        self.grad_clip = grad_clip
        self.use_amp = use_amp and device == "cuda"

        self.optimizer = AdamW(
            model.parameters(),
            lr=lr,
            betas=(0.9, 0.98),
            eps=1e-9,
        )

        def lr_lambda(step: int) -> float:
            if step == 0:
                return 1e-10
            return min(1.0, step**(-0.5) * warmup_steps**0.5)

        self.scheduler = LambdaLR(self.optimizer, lr_lambda)

        self.criterion = nn.CrossEntropyLoss(
            ignore_index=0,
            label_smoothing=label_smoothing,
        )

        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None
        self.best_val_loss = float("inf")

    def _train_epoch(self, epoch: int) -> float:
        """Train one epoch with progress tracking."""
        self.model.train()
        total_loss = 0.0
        total_samples = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task(f"[bold cyan]Epoch {epoch}[/bold cyan]", total=len(self.train_loader))

            for batch_idx, batch in enumerate(self.train_loader):
                src = batch["src"].to(self.device)
                tgt = batch["tgt"].to(self.device)

                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]

                if self.use_amp:
                    with torch.amp.autocast('cuda'):
                        logits = self.model(src, tgt_input)
                        loss = self.criterion(
                            logits.reshape(-1, logits.size(-1)),
                            tgt_output.reshape(-1),
                        )
                        loss = loss / self.accum_steps

                    self.scaler.scale(loss).backward()
                else:
                    logits = self.model(src, tgt_input)
                    loss = self.criterion(
                        logits.reshape(-1, logits.size(-1)),
                        tgt_output.reshape(-1),
                    )
                    loss = loss / self.accum_steps
                    loss.backward()

                total_loss += loss.item() * self.accum_steps
                total_samples += src.size(0)

                if (batch_idx + 1) % self.accum_steps == 0:
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

                    if self.use_amp:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()

                    self.optimizer.zero_grad()
                    self.scheduler.step()

                progress.update(task, advance=1, description=f"[bold cyan]Epoch {epoch}[/bold cyan] • [yellow]Loss: {loss.item():.4f}[/yellow]")

        return total_loss / total_samples

    def _validate(self) -> Tuple[float, float]:
        """Validate on validation set with progress tracking."""
        self.model.eval()
        total_loss = 0.0
        total_acc = 0.0
        total_samples = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[cyan]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("[cyan]Validating[/cyan]", total=len(self.val_loader))

            with torch.no_grad():
                for batch in self.val_loader:
                    src = batch["src"].to(self.device)
                    tgt = batch["tgt"].to(self.device)

                    tgt_input = tgt[:, :-1]
                    tgt_output = tgt[:, 1:]

                    logits = self.model(src, tgt_input)
                    loss = self.criterion(
                        logits.reshape(-1, logits.size(-1)),
                        tgt_output.reshape(-1),
                    )

                    total_loss += loss.item() * src.size(0)

                    preds = torch.argmax(logits, dim=-1)
                    mask = tgt_output != 0
                    acc = (preds == tgt_output).float() * mask.float()
                    total_acc += acc.sum().item()
                    total_samples += mask.sum().item()

                    progress.update(task, advance=1)

        avg_loss = total_loss / len(self.val_loader.dataset)
        avg_acc = total_acc / max(total_samples, 1)

        return avg_loss, avg_acc

    def fit(self, epochs: int, resume_from: Optional[str] = None):
        """Train for given number of epochs with fancy logging."""
        # Header
        console.print(
            Panel(
                "[bold cyan]🧠 Sanskrit Sandhi-Viccheda Transformer[/bold cyan]\n"
                "[dim]Training started • Mixed precision enabled[/dim]",
                border_style="cyan",
                expand=False,
            )
        )

        start_epoch = 0
        if resume_from:
            checkpoint = torch.load(resume_from, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state"])
            start_epoch = checkpoint.get("epoch", 0) + 1
            self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
            console.print(f"\n[yellow]↻ Resumed from epoch {start_epoch}[/yellow]\n")

        epoch_results = []

        for epoch in range(start_epoch, epochs):
            console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
            console.print(f"[bold cyan]Epoch {epoch + 1}/{epochs}[/bold cyan]")
            console.print(f"[bold cyan]{'='*60}[/bold cyan]\n")

            train_loss = self._train_epoch(epoch)
            val_loss, val_acc = self._validate()

            # Create results table
            table = Table(title=f"[bold]Epoch {epoch} Results[/bold]", show_header=True, header_style="bold magenta")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="bold green")
            table.add_row("Train Loss", f"{train_loss:.4f}")
            table.add_row("Val Loss", f"{val_loss:.4f}")
            table.add_row("Val Accuracy", f"{val_acc:.4f}")

            console.print(table)

            epoch_results.append((epoch, train_loss, val_loss, val_acc))

            # Log metrics for visualization
            self._log_metrics(epoch, train_loss, val_loss, val_acc)

            # Check for improvement
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save_checkpoint(epoch, is_best=True)
                console.print("\n[bold green]✨ New best model! Checkpoint saved.[/bold green]\n")
            else:
                self._save_checkpoint(epoch, is_best=False)
                console.print(f"\n[dim]Checkpoint saved (best loss so far: {self.best_val_loss:.4f})[/dim]\n")

        # Final summary
        console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
        console.print(f"[bold cyan]Training Complete! 🎉[/bold cyan]")
        console.print(f"[bold cyan]{'='*60}[/bold cyan]\n")

        summary_table = Table(title="[bold]Training Summary[/bold]", show_header=True, header_style="bold magenta")
        summary_table.add_column("Epoch", style="cyan")
        summary_table.add_column("Train Loss", style="yellow")
        summary_table.add_column("Val Loss", style="yellow")
        summary_table.add_column("Val Acc", style="green")

        for epoch, train_loss, val_loss, val_acc in epoch_results[-5:]:  # Show last 5 epochs
            status = "[green]✨[/green]" if val_loss == self.best_val_loss else " "
            summary_table.add_row(
                f"{epoch} {status}",
                f"{train_loss:.4f}",
                f"{val_loss:.4f}",
                f"{val_acc:.4f}",
            )

        console.print(summary_table)

        best_checkpoint = Path(self.checkpoint_dir) / "checkpoint_best.pt"
        console.print(
            Panel(
                f"[bold green]✓ Best checkpoint saved![/bold green]\n"
                f"[cyan]Path:[/cyan] [bold]{best_checkpoint}[/bold]\n"
                f"[cyan]Best Val Loss:[/cyan] [bold]{self.best_val_loss:.4f}[/bold]",
                border_style="green",
                expand=False,
            )
        )

    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save training checkpoint.

        Args:
            epoch: Current epoch
            is_best: Whether this is the best checkpoint so far
        """
        checkpoint = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
        }

        if is_best:
            path = self.checkpoint_dir / "checkpoint_best.pt"
        else:
            path = self.checkpoint_dir / f"checkpoint_epoch{epoch}.pt"

        torch.save(checkpoint, path)

    def _log_metrics(self, epoch: int, train_loss: float, val_loss: float, val_acc: float):
        """Log metrics to JSON file for visualization.

        Args:
            epoch: Current epoch
            train_loss: Training loss
            val_loss: Validation loss
            val_acc: Validation accuracy
        """
        metrics_path = self.checkpoint_dir / "metrics.json"

        if metrics_path.exists():
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
        else:
            metrics = {"epochs": [], "train_loss": [], "val_loss": [], "val_acc": []}

        metrics["epochs"].append(epoch)
        metrics["train_loss"].append(float(train_loss))
        metrics["val_loss"].append(float(val_loss))
        metrics["val_acc"].append(float(val_acc))

        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
