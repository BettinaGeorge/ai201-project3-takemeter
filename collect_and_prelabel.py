# TakeMeter — Data Collection + Pre-labeling Script
# Run this in Google Colab (or locally with Python 3)
#
# What it does:
#   1. Fetches posts from r/contentcreation and r/ContentCreators via Reddit RSS feeds
#      (no API key or account required)
#   2. Filters out question/help posts that don't fit the label taxonomy
#   3. Pre-labels each post using Groq (llama-3.3-70b-versatile) with your label definitions
#   4. Saves everything to a CSV: text, label, notes, subreddit, url
#
# YOU MUST review and correct every label before using the CSV for training.
# Skim-reviewing defeats the purpose and produces noisy training data.

# ─── INSTALL DEPENDENCIES ─────────────────────────────────────────────────────
# Uncomment and run this cell first if in Colab:
# !pip install groq feedparser -q

import feedparser
import requests
import time
import csv
import re
from html import unescape
from groq import Groq

# ─── CONFIG ───────────────────────────────────────────────────────────────────

GROQ_API_KEY = ""  # Paste your Groq API key here, or use Colab Secrets (see below)

# To use Colab Secrets instead:
# from google.colab import userdata
# GROQ_API_KEY = userdata.get("GROQ_API_KEY")

SUBREDDITS = ["contentcreation", "ContentCreators", "creators", "NewTubers"]
OUTPUT_FILE = "takemeter_prelabeled.csv"

# RSS feed sorts + time windows — pulls from multiple to maximise post count
RSS_FEEDS = [
    ("top", "?t=month"),
    ("top", "?t=year"),
    ("top", "?t=all"),
    ("hot", ""),
    ("new", ""),
]

DELAY_BETWEEN_REQUESTS = 4  # seconds — avoids 429 rate limiting

# Keywords that suggest a post is a question/help request — skip these
SKIP_PATTERNS = [
    "how do i", "how to", "any tips", "any advice", "help me", "need help",
    "i need help", "where can i", "what should i", "should i", "can someone",
    "looking for", "recommendations", "suggest", "advice needed", "advice for",
    "how can i", "help!", "??", "???",
]

# ─── LABEL DEFINITIONS (used in the Groq prompt) ──────────────────────────────

LABEL_DEFINITIONS = """
You are a text classifier for an online content creator community. Classify each post into exactly one of these three labels:

analysis — the post makes a structured argument about content creation strategy, platform behavior, or the industry, with specific reasoning or evidence. The point could stand on its own even without the author's personal feelings.

hot_take — a bold, confident opinion about content creation stated with little or no supporting argument. The claim might be true, but the post asserts rather than reasons.

experience — a personal story or account from someone's creator journey. The post is primarily about what happened to them, not making a general argument.

Decision rule for edge cases: if a post shares a personal story AND makes a broader argument, ask what the PRIMARY purpose is. If it's mostly storytelling → experience. If it's mostly arguing a general point → analysis.

Respond with ONLY one word: analysis, hot_take, or experience. No explanation.
"""

# ─── FETCH POSTS VIA RSS ──────────────────────────────────────────────────────

def clean_html(text):
    """Strip HTML tags and unescape HTML entities."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def fetch_via_rss(subreddit, sort="top", params=""):
    """Fetch posts from a subreddit using its public RSS feed."""
    url = f"https://www.reddit.com/r/{subreddit}/{sort}/.rss{params}&limit=100"
    headers = {"User-Agent": "TakeMeter/1.0 (academic NLP project)"}

    try:
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code == 429:
            print(f"    Rate limited (429) — waiting 10s before continuing...")
            time.sleep(10)
            return []
        if response.status_code != 200:
            print(f"    r/{subreddit} {sort}{params} returned {response.status_code}")
            return []

        feed = feedparser.parse(response.text)
        posts = []

        for entry in feed.entries:
            title = entry.get("title", "").strip()
            summary = clean_html(entry.get("summary", ""))
            link = entry.get("link", "")

            if summary and len(summary) > len(title) + 20:
                full_text = f"{title}\n\n{summary[:1500]}"
            else:
                full_text = title

            posts.append({
                "title": title,
                "text": full_text,
                "subreddit": subreddit,
                "url": link,
            })

        return posts

    except Exception as e:
        print(f"    Error: {e}")
        return []


def is_question_post(title):
    """Return True if the post looks like a question/help request."""
    title_lower = title.lower()
    for pattern in SKIP_PATTERNS:
        if pattern in title_lower:
            return True
    # Also skip very short titles (likely low-content posts)
    if len(title.split()) < 4:
        return True
    return False

def fetch_all_posts(subreddits, feeds):
    """Fetch posts from all subreddits and feed combos, deduplicate by title."""
    all_posts = []
    seen_titles = set()

    for subreddit in subreddits:
        for sort, params in feeds:
            label = f"{sort}{params}" if params else sort
            print(f"  r/{subreddit} ({label})...")
            posts = fetch_via_rss(subreddit, sort, params)
            new = 0
            for p in posts:
                if p["title"] not in seen_titles:
                    seen_titles.add(p["title"])
                    all_posts.append(p)
                    new += 1
            print(f"    +{new} new posts (total: {len(all_posts)})")
            time.sleep(DELAY_BETWEEN_REQUESTS)

    return all_posts


# ─── PRE-LABEL WITH GROQ ──────────────────────────────────────────────────────

def prelabel_post(client, text):
    """Send a post to Groq and get back a label."""
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": LABEL_DEFINITIONS},
                {"role": "user", "content": f"Post:\n{text[:1000]}"},  # cap at 1000 chars
            ],
            max_tokens=10,
            temperature=0,
        )
        label = response.choices[0].message.content.strip().lower()

        # Validate — if the model returns something unexpected, mark for review
        if label not in ["analysis", "hot_take", "experience"]:
            return "REVIEW_NEEDED"
        return label

    except Exception as e:
        print(f"  Groq error: {e}")
        return "REVIEW_NEEDED"


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    if not GROQ_API_KEY:
        print("ERROR: Set your GROQ_API_KEY at the top of the script.")
        return

    client = Groq(api_key=GROQ_API_KEY)

    print("Fetching posts from Reddit RSS feeds (no API key needed)...")
    print("Note: 4s delay between requests to avoid rate limiting — this takes ~5 min\n")
    all_posts = fetch_all_posts(SUBREDDITS, RSS_FEEDS)
    print(f"\nTotal unique posts fetched: {len(all_posts)}")

    # Filter out question/help posts
    filtered = [p for p in all_posts if not is_question_post(p["title"])]
    print(f"After filtering question posts: {len(filtered)} remaining")

    if len(filtered) < 50:
        print("\nWARNING: Fewer than 50 posts after filtering.")
        print("RSS feeds may be rate-limiting. Try running again in a few minutes.")
        return

    # Pre-label with Groq
    print(f"\nPre-labeling {len(filtered)} posts with Groq...")
    results = []
    for i, post in enumerate(filtered):
        label = prelabel_post(client, post["text"])
        results.append({
            "text": post["text"],
            "label": label,   # AI PRE-LABEL — YOU MUST REVIEW EVERY ONE
            "notes": "",      # Add your notes here during review
            "subreddit": post["subreddit"],
            "url": post["url"],
        })
        if (i + 1) % 20 == 0:
            print(f"  Labeled {i + 1}/{len(filtered)}...")
        time.sleep(0.3)

    # Save to CSV
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label", "notes", "subreddit", "url"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nDone! Saved {len(results)} posts to '{OUTPUT_FILE}'")

    # Print label distribution
    from collections import Counter
    dist = Counter(r["label"] for r in results)
    print("\nLabel distribution (AI pre-labels — review before trusting):")
    for label, count in sorted(dist.items()):
        pct = count / len(results) * 100
        print(f"  {label}: {count} ({pct:.0f}%)")

    if max(dist.values()) / len(results) > 0.70:
        print("\nWARNING: One label is over 70% of the dataset.")
        print("After reviewing, balance your dataset before training.")

    print("\n*** IMPORTANT: Open the CSV and read every post. ***")
    print("*** Fix any wrong labels in the 'label' column. ***")
    print("*** Note hard edge cases in the 'notes' column.  ***")


if __name__ == "__main__":
    main()
