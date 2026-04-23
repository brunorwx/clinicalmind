# app/agent/tools/data_tools.py
import json, traceback, sys
from io import StringIO
from langchain_core.tools import tool
from sqlalchemy import text
import pandas as pd
from app.database import async_session

ALLOWED_TABLES = {
    "patients":       "id, trial_id, external_id, arm, enrolled_date, status",
    "adverse_events": "id, patient_id, grade, description, onset_day, resolved",
    "lab_results":    "id, patient_id, test_name, value, unit, collected_at",
}

@tool
async def query_trial_database(sql: str, trial_id: str) -> str:
    """
    Run a read-only SQL SELECT on clinical trial structured data.

    Available tables:
      patients(id, trial_id, external_id, arm, enrolled_date, status)
      adverse_events(id, patient_id, grade, description, onset_day, resolved)
      lab_results(id, patient_id, test_name, value, unit, collected_at)

    Always join through patients to filter by trial_id.

    Args:
        sql:      A SELECT query. No writes permitted.
        trial_id: Used to validate the query scope.
    """
    sql = sql.strip()
    if not sql.upper().startswith("SELECT"):
        return "Error: only SELECT queries are permitted."
    for bad in ["drop", "delete", "insert", "update", "truncate", "--", ";"]:
        if bad in sql.lower():
            return f"Error: forbidden keyword '{bad}' in query."

    async with async_session() as db:
        try:
            rows = (await db.execute(text(sql), {"trial_id": trial_id})).mappings().all()
        except Exception as e:
            return f"SQL error: {e}"

    if not rows:
        return "Query returned no results."
    df = pd.DataFrame([dict(r) for r in rows])
    return df.to_string(index=False, max_rows=100)

@tool
def run_python_analysis(code: str, data_json: str) -> str:
    """
    Execute Python (pandas, statistics) on data from query_trial_database.
    The variable `df` is pre-loaded as a pandas DataFrame from data_json.
    Print results or assign to `result`.

    Args:
        code:      Python code to run.
        data_json: JSON array of records (output of query_trial_database).
    """
    BLOCKED = ["os.", "sys.", "open(", "subprocess", "__import__",
               "eval(", "exec(", "globals(", "locals(", "builtins"]
    for b in BLOCKED:
        if b in code:
            return f"Error: '{b}' is not permitted."
    old_stdout = sys.stdout
    sys.stdout = buf = StringIO()
    try:
        records = json.loads(data_json)
        df = pd.DataFrame(records)                    # noqa: F841
        local_vars: dict = {"df": df, "pd": pd}
        exec(code, {"pd": pd, "json": json}, local_vars)  # noqa: S102
        out = buf.getvalue()
        result = local_vars.get("result", "")
        return (out + str(result)).strip() or "Code ran, no output."
    except Exception:
        return f"Execution error:\n{traceback.format_exc(limit=3)}"
    finally:
        sys.stdout = old_stdout

DATA_TOOLS = [query_trial_database, run_python_analysis]