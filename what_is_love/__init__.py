import re
import praw
import pandas as pd
import logging
from prawcore.exceptions import Forbidden


class RedditScraper:
    def __init__(self, client_id, client_secret, user_agent, subreddits, num_posts=100):
        handler = logging.StreamHandler()
        handler.setLevel(logging.DEBUG)
        for logger_name in ("praw", "prawcore"):
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.DEBUG)
            logger.addHandler(handler)

        self.reddit = praw.Reddit(client_id=client_id, client_secret=client_secret, user_agent=user_agent)
        self.subreddits = subreddits
        self.num_posts = num_posts

    def scrape(self, output_file):
        posts = []
        i, j = 0, 0

        for subreddit in self.subreddits:
            i += 1
            try:
                for submission in self.reddit.subreddit(subreddit).search(query='love', limit=self.num_posts):
                # for submission in self.reddit.subreddit('popular').top(limit=self.num_posts, time_filter='all'):
                    # for submission in self.reddit.subreddit(subreddit).top(limit=self.num_posts, time_filter='all'):
                    if submission.is_self and submission.selftext != '':
                        posts.append({
                            'author': submission.author.name if submission.author else '',
                            'title': submission.title,
                            'score': submission.score,
                            'id': submission.id,
                            'url': submission.url,
                            'num_comments': submission.num_comments,
                            'created': pd.to_datetime(submission.created_utc, unit='s'),
                            'subreddit': subreddit,
                            'selftext': re.sub(r'["\n\t]', ' ', submission.selftext)
                        })
                        j += 1
                        print(f"INFO: Parsed i: {i} and j: {j}")
            except Forbidden:
                print("Something went wrong. Going to next thread")
                continue

        df = pd.DataFrame(posts)
        df.to_csv(output_file, index=False, sep='\t')
