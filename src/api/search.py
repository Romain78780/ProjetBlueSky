import requests

def search_posts(token: str, query: str, max_posts: int, page_limit: int = 100) -> list[dict]:
    """
    Récupère jusqu'à `max_posts` posts contenant `query`
    via l'endpoint app.bsky.feed.searchPosts.
    """
    all_posts = []
    cursor = None
    headers = {"Authorization": f"Bearer {token}"}
    while len(all_posts) < max_posts:
        params = {"q": query, "limit": page_limit}
        if cursor:
            params["cursor"] = cursor
        url = "https://bsky.social/xrpc/app.bsky.feed.searchPosts"
        resp = requests.get(url, headers=headers, params=params)
        if resp.status_code != 200:
            print(f"❌ searchPosts failed [{resp.status_code}]: {resp.text}")
            break
        data = resp.json()
        posts = data.get("posts", [])
        print(f"Page récupérée: {len(posts)} posts (cursor={data.get('cursor')})")
        if not posts:
            break
        all_posts.extend(posts)
        cursor = data.get("cursor")
        if not cursor:
            break
    return all_posts[:max_posts]

def extract_search_tweets(posts: list[dict]) -> list[dict]:
    """
    Transforme la réponse de searchPosts en liste de dict
    simples (uri, handle, text, createdAt).
    """
    tweets = []
    for p in posts:
        rec    = p.get("record", {})
        author = p.get("author", {})
        tweets.append({
            "uri":       p.get("uri", ""),
            "handle":    author.get("handle", ""),
            "text":      rec.get("text", ""),
            "createdAt": rec.get("createdAt", "")
        })
    return tweets
