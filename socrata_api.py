"""
Generic Socrata (data.transportation.gov and most other *.data.gov sites) API
helpers. No FMCSA-specific knowledge lives here -- field names, table URLs and
filters are wired up in the notebook, which is what a reviewer should read to
understand the model. This module is the plumbing underneath it: retries,
counting, and a filter fallback ladder.
"""
import time

import pandas as pd
import requests


def make_session(user_agent):
    session = requests.Session()
    session.headers.update({"User-Agent": user_agent})
    return session


def api_get(session, url, params, retries=3, timeout=30):
    """Retrieve JSON from a Socrata endpoint with retries. Raises on repeated
    failure, printing the response body so a bad query is diagnosable."""
    response = None

    for attempt in range(retries):
        try:
            response = session.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            return response.json()

        except requests.RequestException:
            if attempt == retries - 1:
                print("API request failed.")
                print("URL:", response.url if response is not None else url)
                if response is not None:
                    print("Response:", response.text[:2000])
                raise

            time.sleep(2 ** attempt)


def row_count(session, url, where=None, timeout=30):
    """$select count(1) behind an optional WHERE clause. Returns None (not an
    exception) on failure, so a caller can fall back gracefully."""
    params = {"$select": "count(1)"}
    if where:
        params["$where"] = where

    try:
        result = api_get(session, url, params, retries=1, timeout=timeout)
        if result:
            return int(list(result[0].values())[0])
    except requests.RequestException as exc:
        print(f"Count query failed ({type(exc).__name__}).")

    return None


def fetch_with_where_fallback(session, url, select, order_field, total_wanted,
                              where_attempts, strata=1, timeout=30):
    """
    Try a ladder of WHERE clauses, each weaker than the last, and stratify the
    pull evenly across `order_field` under whichever clause succeeds. A
    query-level failure on one clause (bad SoQL, an unsupported filter on this
    vintage of the table) moves to the next, weaker clause rather than
    raising; only running out of clauses raises. This is what keeps a schema
    quirk on one field from taking down the whole pull.

    where_attempts: list of (label, where_clause_or_None), tried in order.
    Stratifying spreads the pull evenly across the ordering field instead of
    taking the front of the table -- important whenever the population isn't
    already randomly ordered (e.g. a table ordered by ID/registration date).
    """

    def _once(where):
        population = row_count(session, url, where, timeout=timeout)

        if not population or population <= total_wanted:
            params = {"$select": select, "$limit": total_wanted, "$order": order_field}
            if where:
                params["$where"] = where
            return pd.DataFrame(api_get(session, url, params, timeout=timeout)), population

        per_stratum = max(1, total_wanted // strata)
        stride = population // strata
        print(f"Population behind the filter: {population:,}")
        print(f"Sampling {strata} strata x {per_stratum:,} rows, stride {stride:,}.")

        frames = []
        for k in range(strata):
            params = {
                "$select": select, "$limit": per_stratum,
                "$offset": k * stride, "$order": order_field,
            }
            if where:
                params["$where"] = where
            chunk = pd.DataFrame(api_get(session, url, params, timeout=timeout))
            frames.append(chunk)
            print(f"  stratum {k + 1}/{strata}: offset {k * stride:,} -> {len(chunk):,} rows")

        return pd.concat(frames, ignore_index=True), population

    for label, where in where_attempts:
        print(f"\nFilter attempt -- {label}: {where or '(none)'}")
        try:
            df, population = _once(where)
            if population is None:
                print("Population behind the filter: unknown -- pulling without stratification.")
            return df
        except requests.RequestException as exc:
            print(f"  Attempt failed ({type(exc).__name__}) -- trying the next, weaker filter.")

    raise RuntimeError(
        "Pull failed even with no filter at all -- this is a connectivity or outage "
        "issue, not a query problem. Check the URL and network access."
    )


def standardize_join_key(id_series):
    """
    Strip whitespace, drop a trailing '.0' left over from float coercion, and
    map blank/NaN-like strings to true NA. Used everywhere two Socrata tables
    are joined on an identifier, so the key is built identically every time.
    """
    key = (
        id_series.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    )
    return key.mask(key.isin(["nan", "None", ""]), pd.NA)
