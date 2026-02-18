from src.preprocess import preprocess_email
from src.extractive import extractive_summarize
from src.abstractive_input import build_abstractive_input
from src.abstractive import abstractive_rewrite
from src.grounding_filter import remove_ungrounded_lines
from src.evaluation import print_comparison   # ✅ updated

# ---------- Load Email ----------
with open("data/raw_emails.txt", "r", encoding="utf-8") as f:
    text = f.read()

# ---------- Preprocessing ----------
result = preprocess_email(text)

print("\n📌 SUBJECT:\n", result["subject"])

# ---------- Extractive ----------
extractive_summary = extractive_summarize(result["sentences"], top_n=3)

print("\n✨ EXTRACTIVE SUMMARY:\n")
print(extractive_summary)

# ---------- Build Prompt ----------
prompt = build_abstractive_input(
    extractive_summary,
    result["entities"]
)

# ❌ Prompt not printed anymore (clean output)

# ---------- Abstractive Rewrite ----------
abs_summary = abstractive_rewrite(
    prompt,
    result["cleaned_text"],
    extractive_summary,
    result["entities"]
)

# ---------- Grounding Filter ----------
final_summary = remove_ungrounded_lines(
    abs_summary,
    result["cleaned_text"]
)

print("\n🧠 FINAL SUMMARY:\n")
print(final_summary)

# ---------- Evaluation ----------
print_comparison(
    original=result["cleaned_text"],
    extractive_summary=extractive_summary,
    pipeline_summary=final_summary,
    entities=result["entities"]
)
