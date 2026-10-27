from dotenv import load_dotenv

load_dotenv()

from edgar import Company
from edgar.company_reports import TenK

TICKERS = ("JPM", "WFC", "BAC", "C", "GS")
ITEMS = ("Item 1", "Item 1A", "Item 7", "Item 7A", "Item 8")
SHORT_ITEM_CHARS = 2_000
POINTER_PHRASES = ("incorporated by reference", "annual report", "exhibit 13")


def item_texts(tenk: TenK) -> dict[str, str | None]:
    """Text per item; None means the 10-K has no such item."""
    texts = {item: tenk.get(item) for item in ITEMS}
    return {item: None if text is None else str(text) for item, text in texts.items()}


def pointer_phrases_in(text: str) -> list[str]:
    lowered = text.lower()
    return [phrase for phrase in POINTER_PHRASES if phrase in lowered]


def report_bank(ticker: str) -> None:
    filing = Company(ticker).get_filings(form="10-K").latest()
    if filing is None:
        print(f"{ticker}: no 10-K found")
        return

    tenk = filing.obj()
    if not isinstance(tenk, TenK):
        print(f"{ticker}: filing did not parse as a TenK ({type(tenk).__name__})")
        return

    print(f"\n{ticker}  {filing.filing_date}")
    for item, text in item_texts(tenk).items():
        if text is None:
            print(f"  {item:<8} missing")
            continue

        flag = ""
        if len(text) < SHORT_ITEM_CHARS:
            found = pointer_phrases_in(text)
            flag = f"  <-- SHORT, mentions: {', '.join(found) or 'nothing known'}"
        print(f"  {item:<8} {len(text):>9,}{flag}")


def main() -> None:
    for ticker in TICKERS:
        report_bank(ticker)


if __name__ == "__main__":
    main()
