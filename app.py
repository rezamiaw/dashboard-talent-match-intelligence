import math

import altair as alt
import pandas as pd
import requests
import streamlit as st
from sqlalchemy import create_engine, text


@st.cache_resource
def get_engine():
    pg = st.secrets["postgres"]
    return create_engine(pg["url"])


def load_competency_gap():
    sql = """
    WITH joined AS (
        SELECT
            c.employee_id,
            c.pillar_code,
            dcp.pillar_label,
            c.score,
            lp.rating
        FROM competencies_yearly c
        JOIN v_employee_performance_latest lp
          ON c.employee_id = lp.employee_id
        JOIN dim_competency_pillars dcp
          ON c.pillar_code = dcp.pillar_code
    )
    SELECT
        pillar_code,
        pillar_label,
        AVG(CASE WHEN rating = 5 THEN score END)  AS avg_high,
        AVG(CASE WHEN rating <> 5 THEN score END) AS avg_other,
        AVG(CASE WHEN rating = 5 THEN score END)
        - AVG(CASE WHEN rating <> 5 THEN score END) AS diff_high_minus_other
    FROM joined
    GROUP BY pillar_code, pillar_label
    ORDER BY diff_high_minus_other DESC;
    """
    eng = get_engine()
    with eng.connect() as conn:
        df = pd.read_sql(text(sql), conn)
    return df


def load_cognitive_data():
    sql = """
    SELECT
        CASE WHEN lp.rating = 5 THEN 'High (5)'
             ELSE 'Non-High (≠5)'
        END AS perf_group,
        pp.iq,
        pp.gtq,
        pp.tiki,
        pp.pauli,
        pp.faxtor
    FROM profiles_psych pp
    JOIN v_employee_performance_latest lp
      ON pp.employee_id = lp.employee_id;
    """
    eng = get_engine()
    with eng.connect() as conn:
        df = pd.read_sql(text(sql), conn)
    return df


def load_top_strengths(top_n=5):
    sql = """
    WITH latest_perf AS (
        SELECT * FROM v_employee_performance_latest
    ),
    base AS (
        SELECT s.theme, lp.rating
        FROM strengths s
        JOIN latest_perf lp ON s.employee_id = lp.employee_id
        WHERE s.rank BETWEEN 1 AND 5
          AND s.theme IS NOT NULL
    ),
    stats AS (
        SELECT
            theme,
            SUM(CASE WHEN rating = 5 THEN 1 ELSE 0 END)  AS cnt_high,
            SUM(CASE WHEN rating <> 5 THEN 1 ELSE 0 END) AS cnt_other
        FROM base
        GROUP BY theme
    ),
    totals AS (
        SELECT
            SUM(cnt_high)  AS total_high,
            SUM(cnt_other) AS total_other
        FROM stats
    )
    SELECT
        s.theme,
        cnt_high,
        ROUND(cnt_high::numeric / NULLIF(t.total_high,0) * 100, 2) AS pct_high
    FROM stats s
    CROSS JOIN totals t
    WHERE cnt_high > 0
    ORDER BY pct_high DESC
    LIMIT :top_n;
    """
    eng = get_engine()
    with eng.connect() as conn:
        df = pd.read_sql(text(sql), conn, params={"top_n": top_n})
    return df


def load_talent_match():
    sql = "SELECT * FROM v_talent_match;"
    eng = get_engine()
    with eng.connect() as conn:
        df = pd.read_sql(text(sql), conn)
    return df


def load_talent_summary():
    sql = "SELECT * FROM v_talent_summary;"
    eng = get_engine()
    with eng.connect() as conn:
        df = pd.read_sql(text(sql), conn)
    return df


def load_ranked_talent_list():
    """
    Ambil final_match_rate per employee + info organisasi (role, division, dll).
    Kita pakai v_talent_match sebagai sumber final_match_rate,
    lalu join ke tabel employees & dim_* untuk info tambahan.
    """
    sql = """
    WITH ranked AS (
        SELECT
            employee_id,
            fullname,
            final_match_rate
        FROM v_talent_match
        GROUP BY employee_id, fullname, final_match_rate
    )
    SELECT
        r.employee_id,
        r.fullname,
        r.final_match_rate,
        dp.name  AS department,
        dv.name  AS division,
        dr.name  AS directorate,
        dg.name  AS job_level,
        po.name  AS role
    FROM ranked r
    JOIN employees e
      ON r.employee_id = e.employee_id
    LEFT JOIN dim_departments  dp ON e.department_id  = dp.department_id
    LEFT JOIN dim_divisions    dv ON e.division_id    = dv.division_id
    LEFT JOIN dim_directorates dr ON e.directorate_id = dr.directorate_id
    LEFT JOIN dim_grades       dg ON e.grade_id       = dg.grade_id
    LEFT JOIN dim_positions    po ON e.position_id    = po.position_id
    ORDER BY r.final_match_rate DESC;
    """
    eng = get_engine()
    with eng.connect() as conn:
        df = pd.read_sql(text(sql), conn)
    return df


def call_llm(prompt: str, model: str = "x-ai/grok-4.1-fast:free") -> str:
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {st.secrets['openrouter']['api_key']}",
        "HTTP-Referer": "http://localhost:8501",  # ganti ke URL app-mu kalau di-deploy
        "X-Title": "Talent Match Assistant",
        "Content-Type": "application/json",
    }
    data = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an HR analytics assistant. "
                    "You receive talent match scores (TV, TGV, final_match_rate) and must explain them "
                    "in clear, concise business language. Always avoid making definitive hiring decisions; "
                    "frame outputs as recommendations and insights."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.4,
    }

    resp = requests.post(url, headers=headers, json=data, timeout=90)
    if resp.status_code != 200:
        raise Exception(f"{resp.status_code} {resp.text}")
    return resp.json()["choices"][0]["message"]["content"]


st.set_page_config(page_title="Dashboard Company X", layout="wide")

st.title("Step 1 – Success Pattern Discovery Dashboard")
st.caption(
    "Competency • Cognitive • Strengths – berdasarkan rating kinerja (High performer = rating 5)"
)

st.subheader("1. Competency Gap – High vs Non-High Performers")

df_comp = load_competency_gap()

col1, col2 = st.columns([2, 1])

with col1:
    chart_comp = (
        alt.Chart(df_comp)
        .mark_bar()
        .encode(
            x=alt.X("pillar_code:N", title="Pillar"),
            y=alt.Y("diff_high_minus_other:Q", title="Gap Skor (High - Non-High)"),
            tooltip=[
                "pillar_code",
                "pillar_label",
                alt.Tooltip("avg_high:Q", title="Avg High"),
                alt.Tooltip("avg_other:Q", title="Avg Other"),
                alt.Tooltip("diff_high_minus_other:Q", title="Gap"),
            ],
        )
        .properties(height=350)
    )
    st.altair_chart(chart_comp, use_container_width=True)

with col2:
    st.write("**Tabel Ringkas Competency**")
    st.dataframe(df_comp)

st.markdown("---")


st.subheader("2. Cognitive Distribution – High vs Non-High")

df_cog = load_cognitive_data()

df_cog_long = df_cog.melt(
    id_vars="perf_group",
    value_vars=["iq", "gtq", "tiki", "pauli", "faxtor"],
    var_name="variable",
    value_name="value",
)

selected_vars = st.multiselect(
    "Pilih cognitive variables untuk ditampilkan:",
    options=["iq", "gtq", "tiki", "pauli", "faxtor"],
    default=["pauli", "gtq"],
)

df_cog_filtered = df_cog_long[df_cog_long["variable"].isin(selected_vars)]

chart_cog = (
    alt.Chart(df_cog_filtered)
    .mark_boxplot()
    .encode(
        x=alt.X("variable:N", title="Variable"),
        y=alt.Y("value:Q", title="Score"),
        color=alt.Color("perf_group:N", title="Performance Group"),
        tooltip=["perf_group", "variable", "value"],
    )
    .properties(height=350)
)

st.altair_chart(chart_cog, use_container_width=True)

st.markdown("---")

st.subheader("3. Top Strengths Themes – High Performers (Rating = 5)")

top_n_strengths = st.slider(
    "Pilih jumlah Top Strengths yang ingin ditampilkan:",
    min_value=3,
    max_value=15,
    value=5,
    help="Atur berapa banyak tema strengths teratas yang ingin kamu tampilkan di chart.",
)

df_str = load_top_strengths(top_n_strengths)


col3, col4 = st.columns([2, 1])

with col3:
    chart_str = (
        alt.Chart(df_str)
        .mark_bar()
        .encode(
            x=alt.X("theme:N", sort="-y", title="Theme"),
            y=alt.Y("pct_high:Q", title="% High Performer with Theme"),
            tooltip=["theme", "pct_high", "cnt_high"],
        )
        .properties(height=350)
    )
    st.altair_chart(chart_str, use_container_width=True)

with col4:
    st.write("**Data Top Strengths (High Performer)**")
    st.dataframe(df_str)


st.markdown("---")
st.subheader("4. Talent Match – Final Match Rate per Employee")

df_match = load_talent_match()

df_match_emp = (
    df_match[["employee_id", "fullname", "final_match_rate"]]
    .drop_duplicates()
    .sort_values("final_match_rate", ascending=False)
)

st.write("**Ranking Final Match Rate**")
st.dataframe(df_match_emp)

top_n = st.slider("Show Top N employees by match rate", 3, 50, 10)

df_top = df_match_emp.head(top_n)

chart_match = (
    alt.Chart(df_top)
    .mark_bar()
    .encode(
        x=alt.X("fullname:N", sort="-y", title="Employee"),
        y=alt.Y("final_match_rate:Q", title="Final Match Rate"),
        tooltip=["employee_id", "fullname", "final_match_rate"],
    )
    .properties(height=350)
)

st.altair_chart(chart_match, use_container_width=True)

st.markdown("### Detail per TGV & TV")

selected_emp = st.selectbox(
    "Pilih employee untuk lihat detail:",
    df_match_emp["fullname"],
)

emp_id = df_match_emp.loc[df_match_emp["fullname"] == selected_emp, "employee_id"].iloc[
    0
]
df_detail = df_match[df_match["employee_id"] == emp_id]

st.write(f"**Detail Match untuk:** {selected_emp} ({emp_id})")
st.dataframe(df_detail.sort_values(["tgv_name", "tv_name"]))


st.markdown("---")
st.header("Step 2 – Talent Match Analysis")

df_match = load_talent_match()

col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader("🏆 Ranking by Final Match Rate")
    df_rank = (
        df_match[["employee_id", "fullname", "final_match_rate"]]
        .drop_duplicates()
        .sort_values("final_match_rate", ascending=False)
    )
    st.dataframe(df_rank, height=400)

    top_n = st.slider("Show Top N Employees", 3, 30, 10)
    chart_rank = (
        alt.Chart(df_rank.head(top_n))
        .mark_bar()
        .encode(
            x=alt.X("fullname:N", sort="-y", title="Employee"),
            y=alt.Y("final_match_rate:Q", title="Final Match Rate (%)"),
            tooltip=["employee_id", "fullname", "final_match_rate"],
        )
        .properties(height=350)
    )
    st.altair_chart(chart_rank, use_container_width=True)

with col2:
    st.subheader("🔍 Detail per Employee")
    selected_emp = st.selectbox(
        "Pilih Employee untuk Lihat Detail:",
        df_match["fullname"].unique(),
    )

    df_emp = df_match[df_match["fullname"] == selected_emp]
    st.write(f"**{selected_emp}** – breakdown per TGV dan TV")

    chart_tgv = (
        alt.Chart(df_emp)
        .mark_bar()
        .encode(
            x=alt.X("tgv_name:N", title="Talent Group"),
            y=alt.Y("tgv_match_rate:Q", title="TGV Match Rate (%)"),
            color="tgv_name:N",
            tooltip=["tgv_name", "tgv_match_rate"],
        )
        .properties(height=300)
    )

    st.altair_chart(chart_tgv, use_container_width=True)
    st.dataframe(df_emp[["tgv_name", "tv_name", "tv_match_rate"]])


st.markdown("---")
st.header("Step 3 – Role-based JD & Variable Score (AI Assistant)")

df_match_all = load_talent_match()
df_rank = load_ranked_talent_list()

search = st.text_input(
    "Cari nama / employee_id / role (contoh: 'jane', '312', 'Data Analyst')"
)

df_filtered = df_rank.copy()

if search:
    search_lower = search.lower()
    df_filtered = df_filtered[
        df_filtered["fullname"].str.lower().str.contains(search_lower, na=False)
        | df_filtered["employee_id"]
        .astype(str)
        .str.lower()
        .str.contains(search_lower, na=False)
        | df_filtered["role"]
        .fillna("")
        .str.lower()
        .str.contains(search_lower, na=False)
    ]

page_size = 10
total_rows = len(df_filtered)
total_pages = max(1, math.ceil(total_rows / page_size))

col_p1, col_p2 = st.columns([3, 1])
with col_p1:
    st.caption(f"Showing top {min(total_rows, page_size)} of {total_rows} results")
with col_p2:
    page = st.number_input(
        "Page",
        min_value=1,
        max_value=total_pages,
        value=1,
        step=1,
    )

start_idx = (page - 1) * page_size
end_idx = start_idx + page_size

st.dataframe(
    df_filtered.iloc[start_idx:end_idx].reset_index(drop=True)[
        [
            "employee_id",
            "fullname",
            "final_match_rate",
            "role",
            "division",
            "department",
            "directorate",
            "job_level",
        ]
    ],
    use_container_width=True,
)

st.caption(f"Page {page} of {total_pages}")

df_emp_summary = (
    df_match_all[["employee_id", "fullname", "final_match_rate"]]
    .drop_duplicates()
    .sort_values("final_match_rate", ascending=False)
)

st.subheader("Role Summary")

summary_col1, summary_col2 = st.columns(2)

with summary_col1:
    default_role_name = "Data Analyst"
    default_job_level = "Middle"

    st.write("**Role Name:**", default_role_name)
    st.write("**Job Level:**", default_job_level)

with summary_col2:
    st.write("**Selected benchmark employee IDs:**")
    st.write("_(akan muncul setelah kamu pilih di form di bawah)_")

st.markdown("")

st.subheader("1. Role Information")

with st.form("role_form"):
    role_name = st.text_input(
        "Role Name", value=default_role_name, help="Nama jabatan yang ingin dianalisis"
    )
    job_level = st.selectbox(
        "Job Level",
        options=["Junior", "Middle", "Senior", "Lead"],
        index=1,
    )
    role_purpose = st.text_area(
        "Role Purpose",
        placeholder="1–2 kalimat untuk menjelaskan outcome utama role ini",
        help="Contoh: Memastikan analisis data mendukung keputusan bisnis dengan akurat dan tepat waktu.",
    )

    st.markdown("**Employee Benchmarking**")
    st.caption("Pilih maks. 3 karyawan sebagai benchmark high performer untuk role ini")

    df_emp_summary["label"] = df_emp_summary.apply(
        lambda r: f"{r['fullname']} (ID: {r['employee_id']}, Match: {r['final_match_rate']:.1f}%)",
        axis=1,
    )

    selected_labels = st.multiselect(
        "Select Employee Benchmarking (max 3)",
        options=df_emp_summary["label"].tolist(),
        max_selections=3,
    )

    submitted = st.form_submit_button("Generate Job Description & Variable score")

if submitted:
    if len(selected_labels) == 0:
        st.error("Minimal pilih 1 employee sebagai benchmark.")
    elif len(selected_labels) > 3:
        st.error("Maksimal 3 employee sebagai benchmark.")
    else:
        selected_ids = []
        for lbl in selected_labels:
            emp_id = lbl.split("ID:")[1].split(",")[0].strip()
            selected_ids.append(emp_id)
        df_bench = df_match_all[df_match_all["employee_id"].isin(selected_ids)]

        st.markdown("### Benchmark Employees")
        st.write(", ".join(selected_ids))
        st.dataframe(
            df_bench[
                [
                    "employee_id",
                    "fullname",
                    "tgv_name",
                    "tv_name",
                    "tv_match_rate",
                    "tgv_match_rate",
                    "final_match_rate",
                ]
            ].sort_values(["employee_id", "tgv_name", "tv_name"])
        )

        grouped = []
        for emp_id in selected_ids:
            sub = df_bench[df_bench["employee_id"] == emp_id]
            grouped.append(
                {
                    "employee_id": emp_id,
                    "fullname": sub["fullname"].iloc[0],
                    "final_match_rate": float(sub["final_match_rate"].iloc[0]),
                    "tgv": (
                        sub[["tgv_name", "tgv_match_rate"]]
                        .drop_duplicates()
                        .to_dict(orient="records")
                    ),
                    "tv": (
                        sub[["tgv_name", "tv_name", "tv_match_rate"]].to_dict(
                            orient="records"
                        )
                    ),
                }
            )

        context = "Role information:\n"
        context += f"- Role Name: {role_name}\n"
        context += f"- Job Level: {job_level}\n"
        if role_purpose:
            context += f"- Role Purpose: {role_purpose}\n"
        context += "\nBenchmark employees:\n\n"

        for g in grouped:
            context += f"Nama: {g['fullname']} (ID: {g['employee_id']})\n"
            context += f"Final match rate: {g['final_match_rate']:.2f}\n"
            context += "TGV summary:\n"
            for t in g["tgv"]:
                context += f"  - {t['tgv_name']}: {t['tgv_match_rate']:.2f}\n"
            context += "Key TV (variables):\n"
            for v in g["tv"]:
                context += (
                    f"  - [{v['tgv_name']}] {v['tv_name']}: {v['tv_match_rate']:.2f}\n"
                )
            context += "\n"

        prompt = (
            "Kamu adalah HR analytics assistant.\n"
            "Gunakan informasi role dan benchmark employees di bawah ini untuk:\n"
            "1) Menyusun job description singkat (3–5 bullet) untuk role tersebut.\n"
            "2) Menyusun daftar key variables (kompetensi, cognitive, strengths) beserta bobot / pentingnya.\n"
            "3) Jelaskan secara singkat kenapa benchmark employees ini relevan.\n\n"
            f"{context}\n\n"
            "Jawab dalam bahasa Indonesia dengan format:\n"
            "- Role Purpose & Key Outcomes\n"
            "- Job Description (bullet points)\n"
            "- Key Variables & Penjelasan singkat\n"
            "- Catatan tambahan bagi HR / hiring manager."
        )

        with st.spinner(
            "Menghasilkan Job Description & Variable score dengan Grok 4.1 Fast..."
        ):
            try:
                ai_result = call_llm(prompt)
                st.markdown("### Generated Job Description & Variable Score")
                st.write(ai_result)

                job_details_prompt = (
                    "Gunakan konteks berikut untuk membuat rincian Job Details yang terstruktur.\n\n"
                    f"{context}\n\n"
                    "Role Name: " + role_name + "\n"
                    "Job Level: " + job_level + "\n"
                    "Role Purpose: " + (role_purpose or "-") + "\n\n"
                    "Buat Job Details untuk role di atas dengan kategori:\n"
                    "1) Key Responsibilities\n"
                    "2) Work Inputs\n"
                    "3) Work Outputs\n"
                    "4) Qualifications\n"
                    "5) Competencies\n\n"
                    "Format jawaban:\n"
                    "## Key Responsibilities\n- ...\n\n"
                    "## Work Inputs\n- ...\n\n"
                    "## Work Outputs\n- ...\n\n"
                    "## Qualifications\n- ...\n\n"
                    "## Competencies\n- ..."
                )

                ai_job_details = call_llm(job_details_prompt)

                st.markdown("### Job Details (AI Suggested)")
                st.markdown(ai_job_details)

            except Exception as e:
                st.error(f"Error memanggil LLM: {e}")
