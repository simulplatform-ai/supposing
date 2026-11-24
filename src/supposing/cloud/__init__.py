"""Client for the Supposing cloud API. Talks to the API over HTTPS only.

    from supposing.cloud import Client
    sp = Client()                      # SUPPOSING_API_KEY, or the key saved by `supposing login`
    ds = sp.datasets.push("./proposals")
    run = sp.runs.create(ds.id, "Which proposal is most fundable?")
    sp.runs.wait(run.id)
    print(sp.runs.results(run.id).summary)
"""
from .client import AsyncClient, Client, run_body
from .config import DEFAULT_API_URL, config_path, read_config, remove_config, resolve, write_config
from .errors import CloudError
from .folder import read_folder
from .judges import DEFAULT_JUDGE, JUDGE_HELP, LEVELS, JudgeError, api_judge, parse_judge
from .models import (
    Billing,
    Checkout,
    Comparison,
    Dataset,
    Estimate,
    Item,
    ItemSummary,
    LeaderboardEntry,
    LedgerEntry,
    Me,
    Page,
    PriceTier,
    Pricing,
    Progress,
    Results,
    Run,
    RunEvent,
)
from .records import RecordsError, read_records

__all__ = [
    "DEFAULT_API_URL", "DEFAULT_JUDGE", "JUDGE_HELP", "LEVELS", "JudgeError", "api_judge", "parse_judge",
    "AsyncClient", "Billing", "Checkout", "LedgerEntry", "PriceTier", "Pricing", "Client", "CloudError", "Comparison", "Dataset", "Estimate", "Item",
    "ItemSummary", "LeaderboardEntry", "Me", "Page", "Progress", "Results", "Run", "RunEvent", "config_path", "read_config",
    "RecordsError", "read_folder", "read_records", "remove_config", "resolve", "run_body", "write_config",
]
