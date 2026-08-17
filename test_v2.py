"""
test_v2.py
----------
Smoke-tests for dataops-platform-v2 — covers all original rules
plus the new type_overrides (Phase 0) behaviour.
Run with: python test_v2.py
"""
import sys
sys.path.insert(0, ".")

import pandas as pd
from core_engine import apply_transformation_rules, validate_dataframe

PASS = "\033[92mPASSED\033[0m"
FAIL = "\033[91mFAILED\033[0m"
errors = []

def check(name, condition, msg=""):
    if condition:
        print(f"  {PASS}  {name}")
    else:
        print(f"  {FAIL}  {name}: {msg}")
        errors.append(name)

# ===========================================================================
# --- Original rule tests (regression) ---
# ===========================================================================

df = pd.DataFrame({
    "Email":  ["  alice@example.com ", "BOB@test.com", None],
    "Name":   ["  john doe  ", "jane    smith", "CHARLIE"],
    "Sales":  [100.0, None, 250.0],
    "Signup": ["2024-01-05", "05/20/2023", "not-a-date"],
})
rules = {
    "Email": "Trim Whitespace",
    "Name":  "Title Case",
    "Sales": "Fill Blanks with Median",
    "Signup": "Standardize Dates",
}
cleaned, meta = apply_transformation_rules(df, rules)
check("R-Trim Whitespace", cleaned["Email"].iloc[0] == "alice@example.com")
check("R-Title Case + collapse spaces", cleaned["Name"].iloc[1] == "Jane Smith", repr(cleaned["Name"].iloc[1]))
check("R-Fill Blanks with Median", cleaned["Sales"].iloc[1] == 175.0)
check("R-Standardize Dates", cleaned["Signup"].iloc[0] == "2024-01-05")
check("R-blanks_filled", meta["blanks_filled"] == 1)
check("R-date_parse_failures includes None+bad", meta["date_parse_failures"].get("Signup", 0) == 2)

cleaned2, meta2 = apply_transformation_rules(
    pd.DataFrame({"ID": ["A ", "A ", "B", "B"], "Val": [1, 1, 3, 3]}),
    {"ID": "Remove Duplicates"},
)
check("R-Remove Duplicates", meta2["duplicates_removed"] == 2)

_, meta3 = apply_transformation_rules(
    pd.DataFrame({"Name": ["Alice", None, "Bob"]}),
    {"Name": "Fill Blanks with Median"},
)
check("R-Median-on-text skipped", any("Skipped" in l for l in meta3["changes_applied"]))

try:
    validate_dataframe(pd.DataFrame())
    check("R-validate empty raises", False)
except ValueError:
    check("R-validate empty raises", True)

df5 = pd.DataFrame({"Notes": ["Hello, World! #1", "foo@bar.baz"]})
cleaned5, _ = apply_transformation_rules(df5, {"Notes": "Remove Special Characters"})
check("R-Remove Special Characters", cleaned5["Notes"].iloc[0] == "Hello World 1")

_, meta6 = apply_transformation_rules(
    pd.DataFrame({"Qty": [1, 2, 3]}), {"Qty": "UPPERCASE"}
)
check("R-UPPERCASE skipped on numeric", any("Skipped" in l for l in meta6["changes_applied"]))

cleaned7, meta7 = apply_transformation_rules(
    pd.DataFrame({"Revenue": [10.0, None, None, 50.0]}), {"Revenue": "Fill Blanks with 0"}
)
check("R-Fill Blanks with 0", cleaned7["Revenue"].isna().sum() == 0)
check("R-blanks_filled=2", meta7["blanks_filled"] == 2)

df8 = pd.DataFrame({"First": ["John","John","Jane"],"Last": ["Doe","Doe","Smith"],"Score":[90,90,85]})
_, meta8 = apply_transformation_rules(df8, {"First": "Remove Duplicates", "Last": "Remove Duplicates"})
check("R-Compound dedup", meta8["duplicates_removed"] == 1)

# ===========================================================================
# --- NEW: Phase 0 type_overrides tests ---
# ===========================================================================
print()
print("  -- Type override tests --")

# T-1: Text override preserves leading zeros
df_id = pd.DataFrame({"NationalID": [7, 42, 100], "Name": ["Alice", "Bob", "Carol"]})
cleaned_t1, meta_t1 = apply_transformation_rules(
    df_id, {}, type_overrides={"NationalID": "Text"}
)
check("T-Text cast logs change",
      any("Cast 'NationalID' to Text" in l for l in meta_t1["changes_applied"]))
check("T-dtype is object after Text cast",
      cleaned_t1["NationalID"].dtype == object)
# With a string-typed column, the integers become "7", "42", "100"
check("T-int values stringified correctly",
      cleaned_t1["NationalID"].iloc[0] == "7")

# T-2: Text override turns genuine NaN back to pd.NA (not the string "nan")
df_na = pd.DataFrame({"ID": [1.0, None, 3.0]})
cleaned_t2, _ = apply_transformation_rules(
    df_na, {}, type_overrides={"ID": "Text"}
)
# The None row should be pd.NA / NaN, not the literal string "nan"
val = cleaned_t2["ID"].iloc[1]
check("T-nan not stringified",
      pd.isna(val),
      f"Got: {val!r}")

# T-3: Numeric override coerces text to float
df_num = pd.DataFrame({"Revenue": ["100", "200.5", "bad", None]})
cleaned_t3, meta_t3 = apply_transformation_rules(
    df_num, {}, type_overrides={"Revenue": "Numeric"}
)
check("T-Numeric cast", cleaned_t3["Revenue"].iloc[0] == 100.0)
check("T-Numeric coerce bad -> NaN", pd.isna(cleaned_t3["Revenue"].iloc[2]))
check("T-Numeric cast logged", any("Cast 'Revenue' to Numeric" in l for l in meta_t3["changes_applied"]))

# T-4: Datetime override
df_dt = pd.DataFrame({"Signup": ["2024-01-01", "not-a-date", None]})
cleaned_t4, meta_t4 = apply_transformation_rules(
    df_dt, {}, type_overrides={"Signup": "Datetime"}
)
check("T-Datetime cast", pd.api.types.is_datetime64_any_dtype(cleaned_t4["Signup"]))
check("T-Datetime coerce bad -> NaT", pd.isna(cleaned_t4["Signup"].iloc[1]))

# T-5: Phase 0 happens BEFORE Phase 1 — rule applied to overridden type
df_p = pd.DataFrame({"NationalID": [7, 7, 42]})
# Cast to Text first, then Remove Duplicates on the text values
cleaned_t5, meta_t5 = apply_transformation_rules(
    df_p,
    {"NationalID": "Remove Duplicates"},
    type_overrides={"NationalID": "Text"},
)
check("T-Phase 0 before Phase 1", meta_t5["duplicates_removed"] == 1,
      str(meta_t5["duplicates_removed"]))

# T-6: Unknown override gracefully skipped
df_u = pd.DataFrame({"X": [1, 2, 3]})
_, meta_u = apply_transformation_rules(
    df_u, {}, type_overrides={"X": "BinaryBlob"}
)
check("T-Unknown override skipped",
      any("Skipped unknown type override" in l for l in meta_u["changes_applied"]))

# T-7: Override referencing a non-existent column is silently ignored
df_absent = pd.DataFrame({"A": [1, 2, 3]})
cleaned_abs, meta_abs = apply_transformation_rules(
    df_absent, {}, type_overrides={"Z_MISSING": "Text"}
)
check("T-Absent column override ignored", len(cleaned_abs) == 3)

# T-8: No disk writes
import os, tempfile
tmp_before = set(os.listdir(tempfile.gettempdir()))
big_df = pd.DataFrame({"A": range(10000), "B": ["x"] * 10000})
apply_transformation_rules(big_df, {"A": "Fill Blanks with 0"}, type_overrides={"B": "Text"})
tmp_after = set(os.listdir(tempfile.gettempdir()))
check("T-No disk writes", tmp_before == tmp_after, f"New files: {tmp_after - tmp_before}")

# ===========================================================================
print()
if errors:
    print(f"  {len(errors)} test(s) FAILED: {errors}")
    sys.exit(1)
else:
    print("  All tests PASSED.")
    sys.exit(0)
