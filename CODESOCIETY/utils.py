from datetime import datetime

trusted_sources = [
    "BBC News",
    "Reuters",
    "The Hindu",
    "Indian Express",
    "NDTV News"
]

def compute_credibility(source):
    if source in trusted_sources:
        return 0.9
    return 0.5

def detect_outbreak(fake_prob, published_time):
    try:
        published = datetime.fromisoformat(published_time.replace("Z", ""))
        minutes_diff = (datetime.utcnow() - published).seconds / 60

        if fake_prob > 0.8 and minutes_diff < 60:
            return True
    except:
        pass

    return False