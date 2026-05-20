import streamlit as st 

from recommender import recommend_talks


st.set_page_config(page_title="TED Talk Recommendation System", layout="centered")

st.title("TED Talk Recommendation System")
st.write(
    "Describe the kind of TED talk you want to watch, and we'll suggest some talks for you."
)


with st.form("ted_recommender_form"):
    query = st.text_area(
        "What kind of TED talk are you looking for?",
        height=120,
        placeholder=(
            "Example: Climate change and its impact on health. "
            "How can we reduce our carbon footprint?"
        ),
    )

    top_k = st.slider(
        "Number of recommendations", min_value=3, max_value=10, value=5, step=1
    )

    submitted = st.form_submit_button("Get Recommendations")


if submitted:
    if not query.strip():
        st.warning("Please enter a description of the TED talk you want.")
    else:
        with st.spinner("Finding the best TED talks for you..."):
            results = recommend_talks(query, top_k=top_k)

        if results.empty:
            st.info("No recommendations found. Try a different description.")
        else:
            st.subheader("Recommended TED Talks")
            for idx, row in results.reset_index(drop=True).iterrows():
                st.markdown(f"### {idx + 1}. {row['title']}")
                st.markdown(f"**Speaker:** {row['main_speaker']}")
                st.write(row["short_description"])

                url = row["url"]
                if isinstance(url, str) and url.strip():
                    st.link_button("Watch on TED.com", url)
                else:
                    st.info("No video URL available for this talk.")

                st.markdown("---")


