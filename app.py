from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
from torch import nn

CHECKPOINT = Path(__file__).with_name("cameron_hek293_cnn.pt")


def make_cnn():
    return nn.Sequential(
        nn.Conv1d(8, 24, 3, padding=1), nn.ReLU(),
        nn.Conv1d(24, 16, 3, padding=1), nn.ReLU(),
        nn.Flatten(), nn.Linear(320, 1), nn.Flatten(0),
    )


def one_hot(sequences):
    bases = np.array([list(seq[:20]) for seq in sequences])
    return np.stack([bases == base for base in "ACGT"], axis=1)


@st.cache_resource
def load_model():
    torch.set_num_threads(1)
    model = make_cnn()
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.eval()


def rank_csv(contents, model):
    rows = pd.read_csv(BytesIO(contents), dtype=str, encoding="utf-8-sig")
    columns = {name.strip().casefold(): name for name in rows.columns}
    if not {"guide_sequence", "target_sequence"} <= columns.keys():
        raise ValueError("Include Guide_sequence and Target_sequence columns.")
    guides = rows[columns["guide_sequence"]].str.strip().str.upper()
    sites = rows[columns["target_sequence"]].str.strip().str.upper()
    valid = guides.str.fullmatch(r"[ACGT]{20}(?:[ACGT]{3})?", na=False)
    valid &= sites.str.fullmatch(r"[ACGT]{23}", na=False)
    valid &= guides.str[:20].ne(sites.str[:20])
    if not valid.any():
        raise ValueError("No valid off-target pairs were found.")
    ranked = pd.DataFrame({
        "input_row": rows.index[valid] + 1,
        "guide": guides[valid].str[:20],
        "candidate": sites[valid],
    })
    scores = []
    with torch.inference_mode():
        for start in range(0, len(ranked), 1024):
            batch = ranked.iloc[start:start + 1024]
            encoded = np.concatenate([one_hot(batch.guide), one_hot(batch.candidate)], axis=1)
            scores.extend(model(torch.tensor(encoded, dtype=torch.float32)).tolist())
    ranked["cnn_score"] = scores
    ranked = ranked.sort_values(["guide", "cnn_score", "input_row"], ascending=[True, False, True])
    ranked.insert(0, "rank", ranked.groupby("guide").cumcount() + 1)
    return ranked, int((~valid).sum())


def inspect_candidate(model, guide, candidate):
    spacer = candidate[:20]
    variants = [spacer[:index] + base + spacer[index + 1:]
                for index in range(20) for base in "ACGT"]
    encoded = np.concatenate([one_hot([guide] * 81), one_hot([spacer, *variants])], axis=1)
    with torch.inference_mode():
        output = model(torch.tensor(encoded, dtype=torch.float32)).numpy()
    changes = (output[1:] - output[0]).reshape(20, 4)
    return pd.DataFrame({
        "Position": np.arange(1, 21),
        "Guide base": list(guide),
        "Candidate base": list(spacer),
        "Alignment": ["Match" if a == b else "Mismatch" for a, b in zip(guide, spacer)],
        **{base: changes[:, index] for index, base in enumerate("ACGT")},
    })


def render_inspection(model, group):
    options = {f"Rank {row.rank} · input row {row.input_row}": row.input_row
               for row in group.itertuples(index=False)}
    choice = st.selectbox("Candidate to inspect", list(options),
                          key=f"candidate_{group.guide.iloc[0]}")
    selected = options[choice]
    candidate = group.loc[group.input_row == selected].iloc[0]
    alignment = inspect_candidate(model, candidate.guide, candidate.candidate)
    left, center, right = st.columns(3)
    left.metric("Rank within guide", f"{candidate['rank']} of {len(group)}")
    center.metric("CNN score", f"{candidate.cnn_score:.3f}")
    right.metric("Mismatches", f"{alignment.Alignment.eq('Mismatch').sum()} of 20")

    st.subheader("Guide and candidate alignment")
    matches = "".join("|" if a == b else "·" for a, b in zip(candidate.guide, candidate.candidate[:20]))
    st.code(f"Guide      {candidate.guide}\n           {matches}\n"
            f"Candidate  {candidate.candidate[:20]}  {candidate.candidate[20:]} (PAM)",
            language=None)
    st.caption("A vertical bar marks a match. A dot marks a mismatch. The CNN reads positions 1–20.")

    st.subheader("How individual bases affect the CNN score")
    st.caption("Each value is the score change when only that position is replaced with the indicated base. "
               "Positive values raise the score. These are model responses to hypothetical edits.")
    st.dataframe(
        alignment.style.format({base: "{:+.3f}" for base in "ACGT"})
        .map(lambda value: "background-color: #e1f1e6; color: #173825" if value > .0001
                  else "background-color: #f8e4de; color: #552a20" if value < -.0001 else "",
                  subset=list("ACGT")),
        hide_index=True,
    )
    position = st.select_slider("Position to explore", options=list(range(1, 21)), value=20, key="inspection_position")
    changes_at_position = alignment.loc[position - 1, list("ACGT")].astype(float)
    st.bar_chart(changes_at_position.rename("CNN score change"), horizontal=True)
    st.download_button("Download base-by-base comparison", alignment.to_csv(index=False),
                       f"editguard_row_{selected}_base_comparison.csv", "text/csv",
                       key="inspection_export")


def main():
    st.set_page_config(page_title="EditGuard", layout="wide")
    st.title("EditGuard")
    st.caption("Rank and inspect CRISPR off-target candidates with the trained Cameron HEK293 CNN.")
    st.write("Upload your CSV, choose a guide, then open **Inspect a candidate** to see how each base affects the CNN score.")
    upload = st.file_uploader("Upload a candidate CSV", type="csv",
                              help="Required columns: Guide_sequence and Target_sequence.")
    if upload is None:
        return
    try:
        model = load_model()
        ranked, skipped = rank_csv(upload.getvalue(), model)
    except (ValueError, OSError, RuntimeError) as exc:
        st.error(str(exc))
        return
    guides = ranked.guide.unique()
    left, right = st.columns(2)
    left.metric("Candidates ranked", f"{len(ranked):,}")
    right.metric("Guides", len(guides))
    if skipped:
        st.warning(f"Skipped {skipped:,} rows with invalid sequences or exact guide matches.")
    selected = st.selectbox("Guide", guides)
    st.caption("Higher CNN scores rank first. Scores are uncalibrated. This view ranks candidates; it does not calculate held-out performance.")
    group = ranked[ranked.guide == selected]
    ranking, inspection = st.tabs(["Ranked candidates", "Inspect a candidate"])
    with ranking:
        st.dataframe(group, hide_index=True)
        st.download_button("Download rankings for all guides", ranked.to_csv(index=False),
                           "editguard_ranked_candidates.csv", "text/csv")
    with inspection:
        render_inspection(model, group)


if __name__ == "__main__":
    main()
