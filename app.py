import csv
from pathlib import Path
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
DATA = ROOT / "sample_data.csv"
OUT = ROOT / "annotated_output.csv"
COLS = ["language", "sentiment", "text_type", "review_status", "annotator_notes"]
OPTIONS = {
    "language": ["— Select —", "English", "Hindi", "Other", "Uncertain"],
    "sentiment": ["— Select —", "Positive", "Negative", "Neutral", "Not applicable", "Uncertain"],
    "text_type": ["— Select —", "Statement", "Question", "Request / Instruction", "Greeting", "Other", "Uncertain"],
    "review_status": ["— Select —", "Reviewed — looks correct", "Needs second review", "Ambiguous", "Skip"],
}

st.set_page_config(page_title="LinguaLabel", page_icon="🌐", layout="wide")

def get_data():
    df = pd.read_csv(DATA, keep_default_na=False)
    for c in COLS:
        if c not in df:
            df[c] = ""
    if OUT.exists():
        saved = pd.read_csv(OUT, keep_default_na=False)
        if "id" in saved:
            saved = saved[["id"] + [c for c in COLS if c in saved]].drop_duplicates("id", keep="last")
            df = df.drop(columns=[c for c in COLS if c in df.columns]).merge(saved, on="id", how="left")
            for c in COLS:
                if c not in df: df[c] = ""
                df[c] = df[c].fillna("")
    return df

if not DATA.exists():
    st.error("sample_data.csv must be in the same folder as app.py.")
    st.stop()
df = get_data()
if df.empty:
    st.warning("Dataset is empty.")
    st.stop()

total = len(df)
processed = int(df["review_status"].astype(str).str.strip().ne("").sum())
view = st.sidebar.radio("Workspace", ["Annotate", "Quality dashboard", "Dataset overview"])
st.sidebar.title("LinguaLabel")
st.sidebar.metric("Total sentences", total)
st.sidebar.metric("Items with review status", processed)
st.sidebar.progress(processed / total if total else 0)

if view == "Quality dashboard":
    st.title("📊 Quality Dashboard")
    st.caption("Live summary of annotations saved on this device.")
    a, b, c, d = st.columns(4)
    a.metric("Total sentences", total)
    b.metric("Reviewed / status set", processed)
    c.metric("Remaining", total - processed)
    d.metric("Completion", f"{processed / total:.0%}" if total else "0%")
    st.progress(processed / total if total else 0)

    st.subheader("Language distribution")
    x = df[df.language.astype(str).str.strip().ne("")].language.value_counts()
    if x.empty: st.info("No language labels saved yet.")
    else: st.bar_chart(x)

    left, right = st.columns(2)
    with left:
        st.subheader("Sentiment labels")
        x = df[df.sentiment.astype(str).str.strip().ne("")].sentiment.value_counts()
        if x.empty: st.info("No sentiment labels saved yet.")
        else: st.bar_chart(x)
    with right:
        st.subheader("Text types")
        x = df[df.text_type.astype(str).str.strip().ne("")].text_type.value_counts()
        if x.empty: st.info("No text-type labels saved yet.")
        else: st.bar_chart(x)

    st.subheader("Quality review queue")
    follow = df[df.review_status.isin(["Needs second review", "Ambiguous", "Skip"])]
    if follow.empty: st.success("No items are currently marked for follow-up.")
    else: st.dataframe(follow[["id","text"] + COLS], use_container_width=True, hide_index=True)

    st.subheader("Review status breakdown")
    x = df[df.review_status.astype(str).str.strip().ne("")].review_status.value_counts()
    if x.empty: st.info("No review statuses have been set.")
    else: st.dataframe(x.rename_axis("Review status").reset_index(name="Count"), hide_index=True, use_container_width=True)
    st.download_button("Download annotated CSV", df.to_csv(index=False).encode("utf-8-sig"),
                       "annotated_output.csv", "text/csv")
    st.stop()

if view == "Dataset overview":
    st.title("📋 Dataset overview")
    st.caption("Search and filter the dataset to find records that need attention.")
    search_text = st.text_input("Search sentence text", placeholder="Type a word or phrase…")
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    with filter_col1:
        language_filter = st.selectbox("Language filter", ["All"] + OPTIONS["language"][1:])
    with filter_col2:
        sentiment_filter = st.selectbox("Sentiment filter", ["All"] + OPTIONS["sentiment"][1:])
    with filter_col3:
        status_filter = st.selectbox("Review status filter", ["All"] + OPTIONS["review_status"][1:])

    filtered = df.copy()
    if search_text.strip():
        filtered = filtered[filtered["text"].astype(str).str.contains(search_text.strip(), case=False, na=False, regex=False)]
    if language_filter != "All":
        filtered = filtered[filtered["language"].astype(str).eq(language_filter)]
    if sentiment_filter != "All":
        filtered = filtered[filtered["sentiment"].astype(str).eq(sentiment_filter)]
    if status_filter != "All":
        filtered = filtered[filtered["review_status"].astype(str).eq(status_filter)]

    st.caption(f"Showing {len(filtered)} of {len(df)} records.")
    st.dataframe(filtered, use_container_width=True, hide_index=True)
    st.download_button("Download filtered CSV", filtered.to_csv(index=False).encode("utf-8-sig"),
                       "lingualabel_filtered.csv", "text/csv")
    st.download_button("Download full dataset CSV", df.to_csv(index=False).encode("utf-8-sig"),
                       "lingualabel_dataset.csv", "text/csv")
    st.stop()

st.title("🌐 LinguaLabel")
st.caption("Multilingual Text Annotation & Quality Review")
st.write("Annotate each sentence using the project guidelines. Labels are entered by a human; the app does not automatically infer or verify the correct answer.")
if "idx" not in st.session_state: st.session_state.idx = 0
idx = max(0, min(st.session_state.idx, total - 1))
st.session_state.idx = idx
row = df.iloc[idx]
st.progress((idx + 1) / total, text=f"Sentence {idx + 1} of {total}")
left, right = st.columns([1.4, 1])
with left:
    st.subheader(f"Item {row['id']}")
    st.markdown("**Text to annotate**")
    st.info(str(row["text"]))
    st.markdown("**Annotation guidelines**")
    st.markdown("- Choose language based on the sentence as written.\n- Use sentiment only when clear; use Not applicable if sentiment does not apply.\n- Choose the text type that best describes the sentence.\n- Use Uncertain or Ambiguous rather than guessing.\n- Add a note when needed.")
with right:
    vals = {}
    for col, label in [("language","Language"),("sentiment","Sentiment"),("text_type","Text type"),("review_status","Quality review status")]:
        options = OPTIONS[col]
        value = str(row.get(col, "")).strip()
        index = options.index(value) if value in options else 0
        vals[col] = st.selectbox(label, options, index=index, key=f"{col}_{row['id']}")
    vals["annotator_notes"] = st.text_area("Annotator notes (optional)", value=str(row.get("annotator_notes","")), key=f"notes_{row['id']}")

def validation_issues(values):
    issues = []
    required = ["language", "sentiment", "text_type", "review_status"]
    for column in required:
        if not str(values.get(column, "")).strip() or values[column] == "— Select —":
            issues.append(f"Choose a value for {column.replace('_', ' ')}.")
    notes = str(values.get("annotator_notes", "")).strip()
    for column in ["language", "sentiment", "text_type"]:
        if values.get(column) == "Uncertain" and not notes:
            issues.append(f"Add an annotator note explaining the uncertain {column.replace('_', ' ')}.")
    if values.get("review_status") == "Reviewed — looks correct":
        uncertain = [c.replace("_", " ") for c in ["language", "sentiment", "text_type"]
                     if values.get(c) in ("Uncertain", "— Select —", "")]
        if uncertain:
            issues.append("Cannot mark as reviewed while labels are missing or uncertain: " + ", ".join(uncertain) + ".")
    if values.get("review_status") in ("Needs second review", "Ambiguous", "Skip") and not notes:
        issues.append("Add a note explaining this follow-up, ambiguous, or skipped item.")
    return issues

def save():
    issues = validation_issues(vals)
    if issues:
        st.error("Please fix these validation issues before saving:")
        for issue in issues:
            st.markdown(f"- {issue}")
        return False
    current = get_data()
    mask = current["id"].astype(str) == str(row["id"])
    for col, value in vals.items():
        current.loc[mask, col] = value
    current.to_csv(OUT, index=False, quoting=csv.QUOTE_MINIMAL, encoding="utf-8-sig")
    st.session_state.msg = f"Saved annotation for item {row['id']}."
    return True

if "msg" in st.session_state:
    st.success(st.session_state.pop("msg"))
issues_now = validation_issues(vals)
if issues_now:
    with st.expander(f"⚠️ Validation check: {len(issues_now)} issue(s)", expanded=False):
        st.caption("Resolve these before saving.")
        for issue in issues_now:
            st.markdown(f"- {issue}")
else:
    st.success("Validation check passed. This annotation is ready to save.")

p, n, s, u = st.columns([1,1,1.3,1.3])
with p:
    if st.button("← Previous", disabled=idx == 0, use_container_width=True):
        if save():
            st.session_state.idx -= 1
            st.rerun()
with n:
    if st.button("Next →", disabled=idx == total-1, use_container_width=True):
        if save():
            st.session_state.idx += 1
            st.rerun()
with s:
    if st.button("Save annotation", type="primary", use_container_width=True):
        if save():
            st.rerun()
with u:
    if st.button("Go to next unreviewed", use_container_width=True):
        if not save():
            st.stop()
        updated = get_data()
        pending = updated.index[updated.review_status.astype(str).str.strip().eq("")].tolist()
        candidates = [i for i in pending if i > idx] or [i for i in pending if i <= idx]
        if candidates: st.session_state.idx = candidates[0]; st.rerun()
        else: st.success("Every item has a review status. Check the dashboard for follow-up items.")
st.caption(f"Annotations are saved locally in {OUT.name}.")
if OUT.exists():
    st.download_button("Download annotated CSV", OUT.read_bytes(), "annotated_output.csv", "text/csv")
else:
    st.info("Save your first annotation to create the output CSV.")
