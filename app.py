"""Youth Prevention Intelligence dashboard (Streamlit).

Run:  streamlit run app.py
Needs UNBOXAI_API_KEY in the environment for feeds that are not cached yet.
"""
import altair as alt
import pandas as pd
import streamlit as st

import influence as I

st.set_page_config(page_title="Content Influence Index", page_icon="📊", layout="wide")

THEME_COLORS = {"Vaping & nicotine": "#7c3aed", "Alcohol": "#dc2626", "Cannabis": "#16a34a",
                "Stress & coping": "#ea580c", "Prevention": "#0284c7"}
BUCKET_COLORS = {"Vaping & nicotine": "#7c3aed", "Alcohol & drinking gear": "#dc2626",
                 "Cannabis & smoking gear": "#16a34a", "Energy drinks": "#ca8a04", "Quit aids": "#0284c7"}
theme_scale = alt.Scale(domain=list(THEME_COLORS), range=list(THEME_COLORS.values()))
bucket_scale = alt.Scale(domain=list(BUCKET_COLORS), range=list(BUCKET_COLORS.values()))

# ---------------------------------------------------------------- sidebar
st.sidebar.title("Content Influence Index")
st.sidebar.caption("Youth prevention intelligence, powered by BehaviorGPT")
profiles = st.sidebar.multiselect("Simulated youth groups", list(I.PROFILES), default=list(I.PROFILES))
if not profiles:
    st.sidebar.warning("Pick at least one group.")
    st.stop()
st.sidebar.info(
    "**Simulated profiles only.** No real youth data is used. Results describe groups, "
    "never individuals.", icon="🛡️")
st.sidebar.caption(f"Catalog: Unbox AI public sample `{I.CATALOG}`. "
                   f"Each score = % of the top {I.TOP_N} products the model predicts the person looks at next.")


@st.cache_data(show_spinner="Asking BehaviorGPT…")
def load(profiles_key: tuple[str, ...]):
    ps = list(profiles_key)
    return I.direct_table(ps), I.breakthrough_table(ps), I.prevention_table(ps)


direct, breakthrough, prevention = load(tuple(profiles))
df = pd.DataFrame(direct)
df["Breaks through a normal feed (pts)"] = [sum(breakthrough[k].values()) / 3 for k in df["key"]]

st.title("Which social media content pushes youth toward substances?")
st.caption("BehaviorGPT, a model trained on 150 billion real user actions, predicts what a person looks at next. "
           "We show it simulated youth feeds and measure how far each content type shifts those predictions "
           "toward vaping, alcohol and cannabis products.")

tab_rank, tab_groups, tab_feed, tab_prev, tab_method = st.tabs(
    ["🏆 Influence ranking", "👥 Who is most affected", "📱 Feed simulator", "🛡️ Prevention lab", "📖 Method & limits"])

# ---------------------------------------------------------------- 1. ranking
with tab_rank:
    risky = df[df["Theme"] != "Prevention"]
    top = risky.iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Most influential content", top["Content"], f"+{top['Influence (pts vs neutral)']:.0f} pts")
    by_theme = risky.groupby("Theme")["Influence (pts vs neutral)"].mean().sort_values(ascending=False)
    c2.metric("Most influential theme", by_theme.index[0], f"+{by_theme.iloc[0]:.0f} pts avg")
    bt = risky.sort_values("Breaks through a normal feed (pts)", ascending=False).iloc[0]
    c3.metric("Breaks through a normal feed most", bt["Content"],
              f"+{bt['Breaks through a normal feed (pts)']:.1f} pts")

    st.subheader("Influence score: shift toward substance products after one post")
    st.caption("Points = percentage of the top-100 predicted next products that are vaping, alcohol or "
               "cannabis items, minus the same share after an everyday post (sports, music, pets…).")
    chart = alt.Chart(df).mark_bar(cornerRadiusEnd=3).encode(
        x=alt.X("Influence (pts vs neutral):Q", title="Influence (points)"),
        y=alt.Y("Content:N", sort="-x", title=None, axis=alt.Axis(labelLimit=320)),
        color=alt.Color("Theme:N", scale=theme_scale),
        tooltip=["Content", "Theme", alt.Tooltip("Influence (pts vs neutral):Q", format="+.1f"),
                 alt.Tooltip("Vaping & nicotine:Q", format=".1f"),
                 alt.Tooltip("Alcohol & drinking gear:Q", format=".1f"),
                 alt.Tooltip("Cannabis & smoking gear:Q", format=".1f"),
                 alt.Tooltip("Quit aids:Q", format=".1f")],
    ).properties(height=480, width="container")
    st.altair_chart(chart)

    st.subheader("What each post leads to")
    long = df.melt(id_vars=["Content"], value_vars=list(BUCKET_COLORS), var_name="Leads to", value_name="Share")
    stack = alt.Chart(long).mark_bar().encode(
        x=alt.X("Share:Q", title="% of top-100 predicted products", stack="zero"),
        y=alt.Y("Content:N", sort=list(df["Content"]), title=None, axis=alt.Axis(labelLimit=320)),
        color=alt.Color("Leads to:N", scale=bucket_scale),
        tooltip=["Content", "Leads to", alt.Tooltip("Share:Q", format=".1f")],
    ).properties(height=480, width="container")
    st.altair_chart(stack)

    st.subheader("Does it break through a normal feed?")
    st.caption("Same content dropped into an everyday feed 1, 2 or 3 times. Most single posts are drowned out "
               "by everyday content. The ones that still shift predictions are the strongest candidates "
               "for platform rules.")
    bt_rows = [{"Content": I.CONTENT_BY_KEY[k].label, "Times shown": f"x{d}", "Shift (pts)": v}
               for k, ds in breakthrough.items() for d, v in ds.items()]
    heat = alt.Chart(pd.DataFrame(bt_rows)).mark_rect().encode(
        x=alt.X("Times shown:N", title="Times shown in feed"),
        y=alt.Y("Content:N", sort=list(df.sort_values("Breaks through a normal feed (pts)", ascending=False)["Content"]),
                title=None, axis=alt.Axis(labelLimit=320)),
        color=alt.Color("Shift (pts):Q", scale=alt.Scale(scheme="oranges", domainMin=0)),
        tooltip=["Content", "Times shown", alt.Tooltip("Shift (pts):Q", format="+.1f")],
    ).properties(height=480, width=360)
    st.altair_chart(heat)

# ---------------------------------------------------------------- 2. groups
with tab_groups:
    st.subheader("Influence by simulated youth group")
    st.caption("Same content, different groups: BehaviorGPT also reads age group and gender, "
               "so the same post can shift each group differently.")
    g = df.melt(id_vars=["Content", "Theme"], value_vars=profiles, var_name="Group", value_name="Influence")
    heat = alt.Chart(g).mark_rect().encode(
        x=alt.X("Group:N", title=None, sort=profiles),
        y=alt.Y("Content:N", sort=list(df["Content"]), title=None, axis=alt.Axis(labelLimit=320)),
        color=alt.Color("Influence:Q", scale=alt.Scale(scheme="purples", domainMin=0), title="Points"),
        tooltip=["Group", "Content", alt.Tooltip("Influence:Q", format="+.1f")],
    ).properties(height=500, width="container")
    text = heat.mark_text(fontSize=11).encode(text=alt.Text("Influence:Q", format=".0f"),
                                              color=alt.value("#111"))
    st.altair_chart(heat + text)
    st.dataframe(
        g.groupby(["Group", "Theme"])["Influence"].mean().unstack().round(1),
        column_config={t: st.column_config.NumberColumn(format="%+.1f") for t in I.THEMES})

# ---------------------------------------------------------------- 3. feed simulator
with tab_feed:
    st.subheader("Build a feed and see what BehaviorGPT predicts next")
    left, right = st.columns([1, 2])
    with left:
        prof = st.selectbox("Simulated youth", list(I.PROFILES))
        options = list(I.NEUTRAL) + list(I.CONTENT_BY_KEY)
        feed = st.multiselect("Feed, in the order seen (oldest first)", options, default=["n_sports", "vape_ad_candy"],
                              format_func=I.label_for)
        st.caption("Tip: add a prevention post at the end and watch the risk drop.")
    if feed:
        try:
            rows = I.predict(tuple(feed), prof)
            I.save_cache()
        except Exception as e:  # e.g. no API key and not cached
            st.error(f"Could not get a prediction: {e}")
            st.stop()
        s = I.shares(rows)
        with left:
            st.metric("Substance products in next-view predictions", f"{s['Risk']:.0f}%")
            sdf = pd.DataFrame({"Leads to": list(BUCKET_COLORS), "Share": [s[b] for b in BUCKET_COLORS]})
            st.altair_chart(alt.Chart(sdf).mark_bar().encode(
                x=alt.X("Share:Q", title="% of top 100", scale=alt.Scale(domain=[0, 100])),
                y=alt.Y("Leads to:N", title=None, sort=list(BUCKET_COLORS)),
                color=alt.Color("Leads to:N", scale=bucket_scale, legend=None),
            ).properties(height=180, width="container"))
        with right:
            st.markdown("**Top 20 products predicted next**")
            for r in rows[:20]:
                c_img, c_txt = st.columns([1, 6])
                if r.get("image"):
                    c_img.image(r["image"], width=56)
                tag = r["bucket"] or "Other"
                color = BUCKET_COLORS.get(tag, "#6b7280")
                c_txt.markdown(f"**{r['rank']}.** {r['name']}  \n"
                               f"<span style='color:{color};font-weight:600'>{tag}</span>",
                               unsafe_allow_html=True)

# ---------------------------------------------------------------- 4. prevention lab
with tab_prev:
    st.subheader("Which prevention message pulls predictions back down?")
    st.caption("Each risky post is followed by a prevention post. Change = risk after prevention "
               "minus risk after the risky post alone, averaged over all risky posts and groups.")
    pdf = pd.DataFrame(prevention)
    c = st.columns(len(pdf))
    for col, (_, r) in zip(c, pdf.iterrows()):
        col.metric(r["Prevention message"], f"{r['Change in risk (pts)']:+.1f} pts risk",
                   f"{r['Change in quit aids (pts)']:+.1f} pts quit aids", delta_color="normal")
    pick = st.selectbox("Look at one prevention message", list(pdf["key"]), format_func=I.label_for)
    pairs = pd.DataFrame(I.prevention_pairs(pick, profiles)).sort_values("Change")
    st.altair_chart(alt.Chart(pairs).mark_bar(cornerRadiusEnd=3).encode(
        x=alt.X("Change:Q", title="Change in substance share (points)"),
        y=alt.Y("Risky post:N", sort="x", title=None, axis=alt.Axis(labelLimit=320)),
        color=alt.condition("datum.Change < 0", alt.value("#0284c7"), alt.value("#dc2626")),
        tooltip=["Risky post", alt.Tooltip("Risk alone:Q", format=".1f"),
                 alt.Tooltip("Risk after prevention:Q", format=".1f"), alt.Tooltip("Change:Q", format="+.1f")],
    ).properties(height=420, width="container"))
    st.info("Messages that *show or name* the product (vapes, quitting vapes) can keep vape products in the "
            "predictions, while messages about a positive identity (sport, training) move them away. "
            "This mirrors a known concern in prevention research and is worth testing with real audiences.",
            icon="💡")

# ---------------------------------------------------------------- 5. method
with tab_method:
    st.markdown(f"""
### How it works
1. **Content library.** {len(I.CONTENT)} social media content types in five themes, each described in words
   (e.g. *"Sponsored Instagram ad: colorful vape device with candy and mango flavors"*).
2. **Simulated youth.** Four groups (teen girls/boys under 18, young women/men 18-25, US). No real people.
3. **BehaviorGPT** reads the feed as a sequence of things the person viewed and ranks the
   {I.TOP_N} products from Unbox AI's public retail catalog they are most likely to look at next.
4. **Scoring.** Each predicted product is sorted into vaping & nicotine, alcohol, cannabis, energy drinks or
   quit aids by its name. Influence = share of substance products minus the share after an everyday post.

### What this is, and isn't
- BehaviorGPT learned from real shopping and engagement behavior, **not from studies of youth substance use**.
  Scores show which content the model associates with moving toward substance products, a signal for
  *what to study and regulate first*, **not proof of causation**.
- Confirm with real population data (e.g. national youth tobacco and drug-use surveys) before policy use.
- Group-level only: the tool never scores or targets an individual young person.
- The product sorter is keyword-based and can miss or mislabel some items.
""")
