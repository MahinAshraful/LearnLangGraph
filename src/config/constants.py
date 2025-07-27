#Google Places API
GOOGLE_PLACES_BASE_URL = "https://maps.googleapis.com/maps/api/place"
GOOGLE_PLACES_PHOTO_BASE_URL = "https://maps.googleapis.com/maps/api/place/photo"
GOOGLE_PLACES_MAX_RESULTS = 60
GOOGLE_PLACES_TYPES = [
    "restaurant", "meal_takeaway", "meal_delivery",
    "cafe", "bar", "bakery", "food"
]

# OpenAI API
OPENAI_BASE_URL = "https://api.openai.com/v1"
OPENAI_EMBEDDING_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072
}
OPENAI_MAX_TOKENS = {
    "gpt-3.5-turbo": 4096,
    "gpt-4": 8192,
    "gpt-4-turbo": 128000
}

# Weather API
WEATHER_BASE_URL = "https://api.openweathermap.org/data/2.5"

# Eventbrite API
EVENTBRITE_BASE_URL = "https://www.eventbriteapi.com/v3"

# =============================================================================
# SEARCH AND RECOMMENDATION CONSTANTS
# =============================================================================

# Distance and location
DEFAULT_SEARCH_RADIUS_KM = 10.0
MAX_SEARCH_RADIUS_KM = 50.0
MIN_SEARCH_RADIUS_KM = 0.5
WALKING_DISTANCE_KM = 1.5
NEARBY_DISTANCE_KM = 5.0

# Quality thresholds
MIN_RESTAURANT_RATING = 1.0
RECOMMENDED_MIN_RATING = 3.5
HIGH_QUALITY_RATING = 4.0
EXCELLENT_RATING = 4.5
MIN_REVIEW_COUNT = 5
RELIABLE_REVIEW_COUNT = 50
POPULAR_REVIEW_COUNT = 200

# Recommendation limits
MIN_RECOMMENDATIONS = 1
DEFAULT_RECOMMENDATIONS = 10
MAX_RECOMMENDATIONS = 50
MAX_CANDIDATES_TO_SCORE = 100

# Scoring constants (aligned with your document)
SCORING_WEIGHTS = {
    "PREFERENCE_MATCH": 0.50,    # 50% - user preference matching
    "CONTEXT_RELEVANCE": 0.30,   # 30% - context like weather, time
    "QUALITY_SCORE": 0.15,       # 15% - restaurant quality metrics
    "BOOST_SCORE": 0.05          # 5% - special preference boosts
}

# Price level mappings
PRICE_LEVEL_RANGES = {
    1: {"min": 5, "max": 15, "symbol": "$", "label": "Budget"},
    2: {"min": 15, "max": 35, "symbol": "$$", "label": "Moderate"},
    3: {"min": 35, "max": 75, "symbol": "$$$", "label": "Expensive"},
    4: {"min": 75, "max": 200, "symbol": "$$$$", "label": "Fine Dining"}
}

# =============================================================================
# TIME AND SCHEDULING CONSTANTS
# =============================================================================

# Meal time ranges (24-hour format)
MEAL_TIMES = {
    "breakfast": {"start": 6, "end": 11},
    "brunch": {"start": 9, "end": 14},
    "lunch": {"start": 11, "end": 16},
    "dinner": {"start": 17, "end": 22},
    "late_night": {"start": 22, "end": 2}
}

# Peak dining hours
PEAK_HOURS = {
    "breakfast": [8, 9],
    "lunch": [12, 13],
    "dinner": [18, 19, 20]
}

# Days of week (for opening hours)
DAYS_OF_WEEK = [
    "monday", "tuesday", "wednesday", "thursday",
    "friday", "saturday", "sunday"
]

# =============================================================================
# CACHING CONSTANTS
# =============================================================================

# Cache key prefixes
CACHE_PREFIXES = {
    "USER_PREFERENCES": "user_prefs",
    "RESTAURANT_DATA": "restaurant",
    "GOOGLE_PLACES": "gplaces",
    "WEATHER_DATA": "weather",
    "QUERY_RESULTS": "query",
    "EMBEDDINGS": "embeddings",
    "SIMILAR_USERS": "similar_users"
}

# Cache TTL (Time To Live) in seconds
CACHE_TTL = {
    "USER_PREFERENCES": 3600,      # 1 hour
    "RESTAURANT_DATA": 1800,       # 30 minutes
    "GOOGLE_PLACES": 900,          # 15 minutes
    "WEATHER_DATA": 600,           # 10 minutes
    "QUERY_RESULTS": 1800,         # 30 minutes
    "EMBEDDINGS": 86400,           # 24 hours
    "SIMILAR_USERS": 3600          # 1 hour
}

# =============================================================================
# VECTOR DATABASE CONSTANTS
# =============================================================================

# Embedding dimensions
EMBEDDING_DIMENSIONS = 1536  # Default for text-embedding-3-large (reduced)
MAX_EMBEDDING_DIMENSIONS = 3072

# Vector search parameters
DEFAULT_SIMILARITY_THRESHOLD = 0.7
MAX_SIMILAR_USERS = 10
DEFAULT_VECTOR_SEARCH_LIMIT = 20

# Collection names
VECTOR_COLLECTIONS = {
    "USER_PREFERENCES": "user_preferences",
    "RESTAURANT_FEATURES": "restaurant_features",
    "QUERY_EMBEDDINGS": "query_embeddings"
}

# =============================================================================
# LLM PROMPT CONSTANTS
# =============================================================================

# System roles
SYSTEM_ROLES = {
    "QUERY_PARSER": "You are an expert restaurant query parser. Extract structured information from natural language restaurant requests.",
    "REASONING_AGENT": "You are a restaurant recommendation expert. Analyze options and provide reasoning for recommendations.",
    "OUTPUT_FORMATTER": "You are a helpful restaurant recommendation assistant. Format recommendations in a clear, engaging way."
}

# Query parsing keywords
CUISINE_KEYWORDS = {
    "italian": ["italian", "pizza", "pasta", "spaghetti", "marinara", "gelato"],
    "chinese": ["chinese", "kung pao", "lo mein", "dim sum", "szechuan"],
    "japanese": ["japanese", "sushi", "ramen", "tempura", "yakitori", "sashimi"],
    "mexican": ["mexican", "tacos", "burritos", "quesadilla", "salsa", "guacamole"],
    "thai": ["thai", "pad thai", "curry", "tom yum", "pad see ew"],
    "indian": ["indian", "curry", "naan", "biryani", "tandoori", "masala"],
    "french": ["french", "bistro", "croissant", "escargot", "coq au vin"],
    "american": ["american", "burger", "steak", "bbq", "mac and cheese"],
    "mediterranean": ["mediterranean", "hummus", "falafel", "gyro", "tzatziki"]
}

PRICE_KEYWORDS = {
    "budget": ["cheap", "budget", "affordable", "inexpensive", "economical"],
    "moderate": ["moderate", "mid-range", "reasonable"],
    "expensive": ["expensive", "nice", "fancy", "pricey"],
    "fine_dining": ["fine dining", "expensive", "luxurious", "gourmet", "high-end"]
}

AMBIANCE_KEYWORDS = {
    "romantic": ["romantic", "date night", "intimate", "cozy"],
    "casual": ["casual", "relaxed", "laid back", "informal"],
    "expensive": ["expensive", "elegant", "sophisticated", "classy"],
    "family_friendly": ["family", "kids", "children", "family-friendly"],
    "business": ["business", "professional", "meeting", "corporate"],
    "trendy": ["trendy", "hip", "modern", "fashionable"],
    "quiet": ["quiet", "peaceful", "calm", "tranquil"],
    "lively": ["lively", "energetic", "vibrant", "bustling"]
}

FEATURE_KEYWORDS = {
    "outdoor_seating": ["outdoor", "patio", "terrace", "sidewalk", "garden"],
    "live_music": ["live music", "band", "jazz", "acoustic", "entertainment"],
    "parking": ["parking", "valet", "garage"],
    "delivery": ["delivery", "takeout", "take out", "to go"],
    "reservations": ["reservations", "booking", "reserve"],
    "wifi": ["wifi", "internet", "wireless"],
    "wheelchair_accessible": ["wheelchair", "accessible", "handicap"]
}

# =============================================================================
# ERROR MESSAGES
# =============================================================================

ERROR_MESSAGES = {
    "NO_RESULTS": "Sorry, I couldn't find any restaurants matching your criteria. Try broadening your search.",
    "API_ERROR": "I'm having trouble accessing restaurant data right now. Please try again in a moment.",
    "INVALID_LOCATION": "I couldn't find that location. Please provide a more specific address or neighborhood.",
    "PARSING_ERROR": "I didn't quite understand your request. Could you please rephrase it?",
    "RATE_LIMIT": "I'm processing a lot of requests right now. Please wait a moment and try again.",
    "NO_USER_PREFS": "I don't have your preferences yet. I'll use general recommendations for now.",
    "INVALID_QUERY": "Please provide a valid restaurant search query.",
    "SERVER_ERROR": "Something went wrong on our end. Please try again."
}

# =============================================================================
# BUSINESS RULES
# =============================================================================

# Minimum data requirements
MIN_USER_ACTIVITIES_FOR_PERSONALIZATION = 3
MIN_RESTAURANTS_FOR_RECOMMENDATION = 5
MIN_SIMILAR_USERS_FOR_COLLABORATIVE = 2

# Quality filters
EXCLUDE_PERMANENTLY_CLOSED = True
EXCLUDE_TEMPORARILY_CLOSED = False  # Still show if user specifically searches
EXCLUDE_LOW_RATING = True  # Exclude restaurants below MIN_RESTAURANT_RATING

# Diversity requirements
MIN_CUISINE_DIVERSITY = 0.3  # At least 30% different cuisines in results
MAX_SAME_CUISINE_PERCENTAGE = 0.6  # Max 60% of results from same cuisine

# Popularity boosts
POPULAR_RESTAURANT_BOOST = 0.1  # 10% boost for popular restaurants
TRENDING_RESTAURANT_BOOST = 0.05  # 5% boost for trending restaurants
NEW_RESTAURANT_BOOST = 0.03  # 3% boost for new restaurants (first 6 months)

# Social signals
FRIEND_RECOMMENDATION_BOOST = 0.15  # 15% boost if friend liked
SIMILAR_USER_BOOST = 0.08  # 8% boost if similar users liked
COMMUNITY_FAVORITE_BOOST = 0.05  # 5% boost for community favorites

# Contextual boosts
WEATHER_APPROPRIATE_BOOST = 0.1  # 10% boost for weather-appropriate venues
TIME_APPROPRIATE_BOOST = 0.05  # 5% boost for time-appropriate venues
OCCASION_MATCH_BOOST = 0.12  # 12% boost for occasion-appropriate venues

# =============================================================================
# PERFORMANCE CONSTANTS
# =============================================================================

# Timeout settings (seconds)
TIMEOUTS = {
    "API_REQUEST": 30,
    "DATABASE_QUERY": 10,
    "VECTOR_SEARCH": 5,
    "LLM_COMPLETION": 60,
    "CACHE_OPERATION": 2
}

# Rate limiting (requests per minute)
RATE_LIMITS = {
    "GOOGLE_PLACES": 100,
    "OPENAI_API": 60,
    "WEATHER_API": 60,
    "EVENTBRITE": 50,
    "USER_REQUESTS": 120  # Per user
}

# Processing limits
MAX_PROCESSING_TIME_MS = 5000  # 5 seconds max processing time
SLOW_QUERY_THRESHOLD_MS = 1000  # 1 second threshold for slow queries
MAX_CONCURRENT_API_CALLS = 10

# Memory limits
MAX_CACHE_SIZE_MB = 512
MAX_VECTOR_CACHE_SIZE = 10000  # Number of embeddings to cache
MAX_QUERY_HISTORY_SIZE = 100  # Per user

# =============================================================================
# MONITORING AND ANALYTICS
# =============================================================================

# Metrics to track
KEY_METRICS = [
    "recommendation_accuracy",
    "user_satisfaction",
    "click_through_rate",
    "booking_conversion_rate",
    "average_processing_time",
    "cache_hit_ratio",
    "api_error_rate"
]

# Success thresholds
SUCCESS_THRESHOLDS = {
    "RECOMMENDATION_ACCURACY": 0.75,  # 75% of recommendations should be relevant
    "USER_SATISFACTION": 4.0,         # Average rating of 4.0/5.0
    "CLICK_THROUGH_RATE": 0.3,        # 30% of recommendations clicked
    "CACHE_HIT_RATIO": 0.8,           # 80% cache hit rate
    "API_ERROR_RATE": 0.05            # Max 5% API error rate
}

# Alert thresholds
ALERT_THRESHOLDS = {
    "HIGH_ERROR_RATE": 0.1,           # Alert if >10% error rate
    "SLOW_RESPONSE": 3000,            # Alert if >3s average response time
    "LOW_CACHE_HIT": 0.5,             # Alert if <50% cache hit rate
    "API_FAILURES": 0.2               # Alert if >20% API failures
}

# =============================================================================
# A/B TESTING CONSTANTS
# =============================================================================

# Experiment configurations
DEFAULT_EXPERIMENT_DURATION_DAYS = 14
MIN_SAMPLE_SIZE_PER_VARIANT = 100
STATISTICAL_SIGNIFICANCE_THRESHOLD = 0.05

# Experiment types
EXPERIMENT_TYPES = [
    "scoring_algorithm",
    "recommendation_strategy",
    "ui_presentation",
    "personalization_level"
]

# =============================================================================
# DATA VALIDATION CONSTANTS
# =============================================================================

# Field validation
MAX_QUERY_LENGTH = 500
MAX_USER_NAME_LENGTH = 100
MAX_RESTAURANT_NAME_LENGTH = 200
MAX_ADDRESS_LENGTH = 500
MAX_REVIEW_TEXT_LENGTH = 2000

# Coordinate bounds (roughly global)
MIN_LATITUDE = -90.0
MAX_LATITUDE = 90.0
MIN_LONGITUDE = -180.0
MAX_LONGITUDE = 180.0

# Rating bounds
MIN_RATING = 1.0
MAX_RATING = 5.0
RATING_PRECISION = 1  # Number of decimal places

# Price bounds (USD)
MIN_PRICE_PER_PERSON = 1
MAX_PRICE_PER_PERSON = 500

# =============================================================================
# FEATURE FLAGS
# =============================================================================

# Default feature states
DEFAULT_FEATURE_FLAGS = {
    "ENABLE_PERSONALIZATION": True,
    "ENABLE_COLLABORATIVE_FILTERING": True,
    "ENABLE_SMART_REASONING": True,
    "ENABLE_WEATHER_CONTEXT": False,  # Not implemented yet
    "ENABLE_EVENT_CONTEXT": False,    # Not implemented yet
    "ENABLE_SOCIAL_FEATURES": False,  # Future feature
    "ENABLE_REAL_TIME_POPULARITY": True,
    "ENABLE_DIETARY_FILTERING": True,
    "ENABLE_A_B_TESTING": True,
    "ENABLE_ANALYTICS": True,
    "ENABLE_CACHING": True
}

# =============================================================================
# CUISINE AND DIETARY MAPPINGS
# =============================================================================

# Cuisine compatibility with dietary restrictions
DIETARY_COMPATIBLE_CUISINES = {
    "vegetarian": ["indian", "mediterranean", "italian", "thai", "mexican"],
    "vegan": ["indian", "mediterranean", "thai", "vietnamese"],
    "gluten_free": ["mexican", "thai", "indian", "mediterranean"],
    "halal": ["middle_eastern", "indian", "mediterranean"],
    "kosher": ["middle_eastern", "mediterranean"],
    "keto": ["american", "italian", "french", "steakhouse"],
    "paleo": ["american", "mediterranean", "steakhouse"]
}

# Cuisine -> typical price level mapping
TYPICAL_CUISINE_PRICE_LEVELS = {
    "fast_food": [1],
    "american": [1, 2, 3],
    "italian": [2, 3],
    "chinese": [1, 2],
    "mexican": [1, 2],
    "thai": [1, 2],
    "japanese": [2, 3, 4],
    "french": [3, 4],
    "indian": [1, 2],
    "mediterranean": [2, 3]
}

# =============================================================================
# LOCATION-SPECIFIC CONSTANTS
# =============================================================================

# Major city coordinates (for testing and defaults)
MAJOR_CITIES = {
    "new_york": {"lat": 40.7128, "lng": -74.0060, "name": "New York, NY"},
    "los_angeles": {"lat": 34.0522, "lng": -118.2437, "name": "Los Angeles, CA"},
    "chicago": {"lat": 41.8781, "lng": -87.6298, "name": "Chicago, IL"},
    "san_francisco": {"lat": 37.7749, "lng": -122.4194, "name": "San Francisco, CA"},
    "miami": {"lat": 25.7617, "lng": -80.1918, "name": "Miami, FL"}
}

# Default location (NYC) for testing
DEFAULT_LOCATION = MAJOR_CITIES["new_york"]

# =============================================================================
# REGEX PATTERNS
# =============================================================================

# Validation patterns
PATTERNS = {
    "EMAIL": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
    "PHONE": r"^\+?1?[-.\s]?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})$",
    "COORDINATES": r"^-?([1-8]?[0-9]\.{1}[0-9]{1,6}$|90\.{1}0{1,6}$),-?([1-9]?[0-9]{1,2}\.{1}[0-9]{1,6}$|1[0-7][0-9]\.{1}[0-9]{1,6}$|180\.{1}0{1,6})$",
    "GOOGLE_PLACE_ID": r"^[A-Za-z0-9_-]{20,}$",
    "UUID": r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
}

# =============================================================================
# HTTP STATUS CODES
# =============================================================================

# Custom status codes for internal use
INTERNAL_STATUS_CODES = {
    "RECOMMENDATION_SUCCESS": 200,
    "PARTIAL_RESULTS": 206,
    "NO_RECOMMENDATIONS": 204,
    "INVALID_QUERY": 400,
    "RATE_LIMITED": 429,
    "API_ERROR": 502,
    "PROCESSING_TIMEOUT": 504
}

# =============================================================================
# VERSION INFORMATION
# =============================================================================

API_VERSION = "1.0.0"
RECOMMENDATION_ENGINE_VERSION = "1.0.0"
EMBEDDING_MODEL_VERSION = "text-embedding-3-large"
CACHE_VERSION = "v1"

# =============================================================================
# DEVELOPMENT AND TESTING
# =============================================================================

# Test data constants
TEST_USER_COUNT = 20
TEST_RESTAURANT_COUNT = 100
TEST_QUERY_COUNT = 50

# Mock data seeds (for consistent testing)
RANDOM_SEED = 42
MOCK_DATA_VERSION = "1.0"

# Debug settings
DEBUG_LOG_LEVEL = "DEBUG"
VERBOSE_SCORING = False  # Set to True for detailed scoring logs
PROFILE_PERFORMANCE = False  # Set to True for performance profiling