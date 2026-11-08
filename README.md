# Content Influence Index

Youth prevention intelligence: which social media content pushes young people toward vaping,
alcohol and cannabis, measured with BehaviorGPT (Unbox AI) on **simulated** youth profiles.

## Run it

1. Save your Unbox AI key once (PowerShell), then open a new terminal:
   `setx UNBOXAI_API_KEY "your-key"`
2. Install: `pip install -r requirements.txt`
3. Precompute results (about 30 s, cached in `results_cache.json`): `python influence.py`
4. Start the dashboard: `streamlit run app.py --server.address localhost`

Cached results let the dashboard show the rankings even without internet. The feed simulator
needs the key for feeds that are not cached yet.

## Files

- `influence.py`: content library, simulated profiles, product sorter, and the three measurements
  (direct influence, breakthrough in a normal feed, prevention effect).
- `app.py`: Streamlit dashboard (ranking, groups, feed simulator, prevention lab, method).

## Headline results (4 simulated groups, top-100 predictions)

- Strongest direct influence: stoner meme (+55 pts), sponsored candy-colored vape ad (+52),
  vape tricks challenge (+47), hookah reel (+46).
- In a normal mixed feed most single posts are drowned out; sponsored vape ads and drinking memes
  still break through.
- After a risky post: a teen-athlete "substance-free" video cuts risk by about 30 pts. An anti-vaping
  PSA cuts it by about 6. A quit-program ad leaves it unchanged.

## Limits

BehaviorGPT learned from shopping and engagement behavior, not youth substance-use studies.
Scores are associations that flag what to study and regulate first, not proof of causation.
Simulated, group-level data only; never score individual young people.
